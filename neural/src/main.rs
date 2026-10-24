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
}

#[derive(Serialize)]
struct Response {
    id: u64,
    promoted: Option<usize>,
    elapsed_ms: f64,
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
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: wime-neural MODEL.safetensors")?;
    let model = SentenceModel::load(&std::fs::read(path)?)?;
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
        serde_json::to_writer(&mut stdout, &evaluate(&mut reranker, &request))?;
        writeln!(stdout)?;
        stdout.flush()?;
    }
    Ok(())
}
