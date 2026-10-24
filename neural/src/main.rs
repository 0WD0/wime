//! Persistent, local JSON-lines adapter. Stdout is exclusively the protocol.
use chinese_ime_lm::{CandidateFacts, Reranker, SentenceModel};
use serde::{Deserialize, Serialize};
use std::io::{self, BufRead, Write};
use std::sync::Arc;
use std::time::Instant;

#[derive(Deserialize)]
struct Candidate {
    text: String,
    end: usize,
    dictionary: bool,
}

#[derive(Deserialize)]
struct Request {
    id: u64,
    context: String,
    syllables: usize,
    candidates: Vec<Candidate>,
    #[serde(default)]
    operation: String,
    #[serde(default)]
    input: String,
    #[serde(default)]
    draft: String,
}

#[derive(Serialize)]
struct Response {
    id: u64,
    promoted: Option<usize>,
    elapsed_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefixes: Option<Vec<String>>,
}

const INPUT: char = '\u{e2e0}';
const OUTPUT: char = '\u{e2e1}';
const KEYS: &str = "abcdefghijklmnopqrstuvwxyz;";

fn prompt(context: &str, input: &str, window: usize) -> Option<String> {
    let keys: Vec<char> = input.chars().collect();
    if keys.is_empty() || !keys.len().is_multiple_of(2) {
        return None;
    }
    let room = window.checked_sub(keys.len() + 4)?.min(64);
    let clean: Vec<char> = context
        .chars()
        .filter(|c| !('\u{e000}'..='\u{f8ff}').contains(c))
        .collect();
    let mut result: String = clean[clean.len().saturating_sub(room)..].iter().collect();
    result.push(INPUT);
    for pair in keys.as_chunks::<2>().0 {
        let first = KEYS.find(pair[0])?;
        let second = KEYS.find(pair[1])?;
        result.push(char::from_u32(0xe000 + (27 * first + second) as u32)?);
    }
    result.push(OUTPUT);
    Some(result)
}

fn verify(reranker: &mut Reranker, request: &Request, conditional: bool) -> Response {
    let start = Instant::now();
    let mut response = Response {
        id: request.id,
        promoted: None,
        elapsed_ms: 0.0,
        score: None,
        prefixes: Some(Vec::new()),
    };
    if !conditional
        || request.draft.chars().count() != request.syllables
        || request.input.len() != 2 * request.syllables
    {
        return response;
    }
    if let Some(prompt) = prompt(
        &request.context,
        &request.input,
        reranker.model().context_length(),
    ) {
        if let Some(distributions) = reranker.verify(&prompt, &request.draft) {
            let vocabulary: Vec<(char, usize)> = ('\u{3400}'..='\u{9fff}')
                .filter_map(|c| reranker.model().token(c).map(|id| (c, id as usize)))
                .collect();
            let mut prefix = String::new();
            let mut score = 0.0;
            let mut edits = Vec::new();
            for (ch, row) in request.draft.chars().zip(distributions) {
                let current = row[reranker.model().token(ch).unwrap_or(1) as usize];
                score += current;
                let mut alternatives: Vec<_> = vocabulary
                    .iter()
                    .copied()
                    .filter(|(c, id)| *c != ch && row[*id] > current + 0.5)
                    .collect();
                alternatives.sort_by(|a, b| row[b.1].total_cmp(&row[a.1]));
                for (character, _) in alternatives.into_iter().take(4) {
                    edits.push(format!("{prefix}{character}"));
                }
                prefix.push(ch);
            }
            edits.truncate(16);
            response.score = Some(score / request.syllables as f32);
            response.prefixes = Some(edits);
        }
    }
    response.elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    response
}

fn evaluate(reranker: &mut Reranker, request: &Request) -> Response {
    let start = Instant::now();
    let texts: Vec<&str> = request.candidates.iter().map(|c| c.text.as_str()).collect();
    // Do not rank a truncated sentence as if the model had seen its ending.
    let in_window = texts
        .iter()
        .all(|s| s.chars().count() < reranker.model().context_length());
    let promoted = if in_window && request.syllables > 0 {
        reranker.best_where(&request.context, &texts, |i| CandidateFacts {
            answers_key: request.candidates[i].end == request.syllables,
            trusted_dictionary_hit: request.candidates[i].dictionary,
        })
    } else {
        None
    };
    Response {
        id: request.id,
        promoted,
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        score: None,
        prefixes: None,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: wime-neural MODEL.safetensors")?;
    let bytes = std::fs::read(path)?;
    let model = SentenceModel::load(&bytes)?;
    // SentenceModel has already validated the safetensors header and geometry.
    let length = u64::from_le_bytes(bytes[..8].try_into()?) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + length])?;
    let conditional = header["__metadata__"]["conditional"] == "wime-ms-pinyin-v1";
    let mut reranker = Reranker::new(Arc::new(model));
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let request: Request = match serde_json::from_str(&line?) {
            Ok(request) => request,
            Err(_) => {
                // No input text in diagnostics. A malformed client request terminates the worker;
                // the input method keeps its traditional candidates on process failure.
                return Err("invalid wime request".into());
            }
        };
        let response = if request.operation == "verify" {
            verify(&mut reranker, &request, conditional)
        } else if conditional {
            // Never apply an input-conditioned model as an unconditional reranker.
            Response {
                id: request.id,
                promoted: None,
                elapsed_ms: 0.0,
                score: None,
                prefixes: None,
            }
        } else {
            evaluate(&mut reranker, &request)
        };
        serde_json::to_writer(&mut stdout, &response)?;
        writeln!(stdout)?;
        stdout.flush()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_preserves_distinct_keys_and_bounds_context() {
        assert_ne!(prompt("", "bmyi", 128), prompt("", "pmyi", 128));
        assert!(prompt("", "bmA", 128).is_none());
        assert!(prompt("", "a", 128).is_none());
        assert!(prompt("", &"aa".repeat(63), 128).is_none());
        assert_eq!(
            prompt(&"甲".repeat(100), "aabb", 128)
                .unwrap()
                .chars()
                .count(),
            68
        );
        assert_eq!(prompt("\u{e2e1}甲", "aa", 128), prompt("甲", "aa", 128));
    }
}
