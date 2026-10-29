"""Register command-line experiments in the same store used by Emacs."""
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import time


def add_arguments(parser):
    parser.add_argument('--experiment', default=os.environ.get('WIME_LAB_EXPERIMENT'),
                        help='experiment directory grouping training and evaluations')
    parser.add_argument('--experiment-name', default=os.environ.get('WIME_LAB_EXPERIMENT_NAME'))
    parser.add_argument('--lab-root', default=os.environ.get('WIME_LAB_ROOT', str(
        Path(os.environ.get('XDG_DATA_HOME', Path.home()/'.local/share'))/'wime')))


def controller():
    path = Path(__file__).resolve().parents[3]/'appkit-apps/wime-lab.el/lab.py'
    spec = importlib.util.spec_from_file_location('wime_lab_controller', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def track(args, kind):
    lab = controller()
    store = lab.Store(args.lab_root)
    out = args.out.resolve()
    directory = Path(args.experiment).resolve() if args.experiment else (
        out.parent if out.name in ('train','training','development','core') or
        any((out.parent/name).exists() for name in ('report.org','plan.json')) else out)
    directory.mkdir(parents=True, exist_ok=True)
    experiment = store.experiment(directory, args.experiment_name)
    key = lab.identity(dict(directory=str(out), kind=kind))[:24]
    activity = dict(id=key, experiment=experiment['id'], name=out.name, kind=kind,
                    directory=str(out), status='running', created=time.time(), pid=os.getpid(),
                    process_token=lab.process_token(os.getpid()))
    store.put('activity', key, activity)

    def checkpoint(status='running'):
        return store.register_training(out, directory, args.experiment_name, status)

    try:
        yield checkpoint
        if kind == 'train':
            registered = checkpoint('completed')
        else:
            registered = (store.import_history_evaluation(out, directory, args.experiment_name)
                          if (out/'results.jsonl').exists() else store.import_evaluation(out, experiment=directory, name=args.experiment_name))
            for entry in registered if isinstance(registered, list) else [registered]:
                entry['experiment'] = experiment['id']
                store.put('evaluation', entry['id'], entry)
                store.link_experiment(experiment, 'evaluations', entry['id'])
        activity.update(status='completed', finished=time.time())
        lab.atomic_json(out/'lab-registration.json', dict(experiment=experiment['id'], activity=key))
    except BaseException as error:
        activity.update(status='failed', error=str(error), finished=time.time())
        if kind == 'train' and (out/'run.json').exists():
            store.register_training(out, directory, args.experiment_name, 'failed', str(error))
        raise
    finally:
        store.put('activity', key, activity)
        store.db.close()
