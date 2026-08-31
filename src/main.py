"""
main.py

Single entry point for the LivestockWaterUse pipeline.

    python main.py                       every stage, every animal
    python main.py --livestock hogs      one animal
    python main.py --stage wcc wc        selected stages
    python main.py --from ratio          resume from a stage onward
    python main.py --dry-run             print the plan, run nothing
    python main.py --list                show the stages and stop

Settings live in config.yaml, not here. A key in a stage's config block
becomes a command line flag on the underlying script -- `n_samples: 50000`
is passed as `--n-samples 50000`, `interpolate: true` as `--interpolate`,
and a key set to null or false is omitted.

--------------------------------------------------------------------------
STRUCTURE
--------------------------------------------------------------------------
    extract     usgs, usda, climate     inputs, independent of each other
    pathway 1   mlr -> wcc -> wc        water consumption per county
    pathway 2   ratio                   consumption-to-withdrawal ratio
    evaluate    validate                WW = WC / ratio, compared with USGS

Stages are ordered by dependency, and requesting one out of order is
allowed -- the ordering is applied automatically, so `--stage wc mlr` runs
mlr first. Each stage declares what it needs, and the run stops with a
clear message if a required input directory is empty rather than failing
somewhere deep inside a script.

Stages are executed by importing the script and calling its main(), not by
shelling out, so that the selected livestock can be applied: each module
exposes a LIVESTOCK constant that is narrowed before the call. Extraction
stages ignore it, since USGS and ERA5 are not animal-specific.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

REPO_DEFAULT = Path(__file__).resolve().parent
LIVESTOCK_ALL = ("dairy_cattle", "beef_cattle", "hogs", "poultry")


@dataclass
class Stage:
    name: str
    group: str
    module: str
    describe: str
    needs: List[str] = field(default_factory=list)   # data dirs that must exist
    produces: str = ""
    per_livestock: bool = False
    # A foundation stage is shared by every user and every animal: the
    # source data and the WCC equations do not depend on which livestock
    # you asked for. These are built once, reused thereafter, and always
    # run over all four animals regardless of --livestock.
    foundation: bool = False


# Dependency order. Anything requested out of order is reordered to this.
STAGES: List[Stage] = [
    Stage("usgs", "foundation", "extract_usgs_livestock",
          "USGS livestock water use, state and county",
          produces="usgs_ls_state_county_1960_2015", foundation=True),
    Stage("usda", "foundation", "usda_livestock",
          "USDA census head counts and herd composition",
          produces="usda_census", foundation=True),
    Stage("climate", "foundation", "era5_climate_GEE",
          "ERA5-Land county and state forcings",
          produces="climate", foundation=True),
    Stage("mlr", "foundation", "wcc_mlr",
          "literature WCC samples and MLR equations",
          produces="wcc_mlr", foundation=True),
    Stage("wcc", "pathway 1", "wcc_ann_downscale",
          "transfer generic WCC to county level",
          needs=["climate", "usda_census", "wcc_mlr"],
          produces="wcc_county", per_livestock=True),
    Stage("wc", "pathway 1", "county_level_WC",
          "county water consumption = WCC x head count",
          needs=["wcc_county", "usda_census"],
          produces="wc_county", per_livestock=True),

    Stage("ratio", "pathway 2", "ratio_ann_downscale",
          "downscale the state WC:WW ratio to counties",
          needs=["usgs_ls_state_county_1960_2015", "climate"],
          produces="ratio_county", per_livestock=True),

    Stage("validate", "evaluate", "validate_against_usgs",
          "WW = WC / ratio, compared with USGS",
          needs=["wc_county", "ratio_county",
                 "usgs_ls_state_county_1960_2015"],
          produces="validation", per_livestock=True),
]

STAGE_BY_NAME = {s.name: s for s in STAGES}
ORDER = {s.name: i for i, s in enumerate(STAGES)}


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    try:
        import yaml
    except ImportError:
        raise SystemExit(
            "PyYAML is required to read config.yaml.\n"
            "    pip install pyyaml")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return cfg


def to_argv(block: Optional[dict]) -> List[str]:
    """
    Turn a config block into command line arguments.

    `n_samples: 50000` -> ['--n-samples', '50000']
    `interpolate: true` -> ['--interpolate']
    `hidden: [64, 32]`  -> ['--hidden', '64', '32']
    null or false       -> omitted
    """
    argv: List[str] = []
    for key, val in (block or {}).items():
        flag = "--" + str(key).replace("_", "-")
        if val is None or val is False:
            continue
        if val is True:
            argv.append(flag)
        elif isinstance(val, (list, tuple)):
            argv.append(flag)
            argv.extend(str(v) for v in val)
        else:
            argv.extend([flag, str(val)])
    return argv


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------

def narrow_livestock(module, livestock: Sequence[str]) -> Optional[str]:
    """
    Restrict a module to the selected animals.

    Each pipeline module keeps its animal list in a module-level constant.
    Narrowing it before calling main() is what makes --livestock work
    without touching the scripts themselves.
    """
    applied = None
    for const in ("LIVESTOCK", "LIVESTOCK_TYPES"):
        if hasattr(module, const):
            setattr(module, const, tuple(livestock))
            applied = const
    return applied


STAMP = ".pipeline.json"


def fingerprint(stage: Stage, cfg: dict, scripts_dir: Path) -> str:
    """
    What the output depends on: the script's source and the stage's config.

    Presence of a directory is not enough to decide a stage can be
    skipped. An output produced by a different version of the script is
    worse than no output at all -- it is reused silently and fails much
    later, somewhere unrelated. Hashing the source and the settings makes
    "already built" mean "built by this code with these settings".
    """
    h = hashlib.sha256()
    src = scripts_dir / f"{stage.module}.py"
    h.update(src.read_bytes() if src.exists() else b"missing")
    h.update(json.dumps(cfg.get(stage.name) or {}, sort_keys=True).encode())
    return h.hexdigest()[:16]


def read_stamp(stage: Stage, data_dir: Path) -> dict:
    p = data_dir / stage.produces / STAMP
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:                              # noqa: BLE001
        return {}


def write_stamp(stage: Stage, data_dir: Path, fp: str,
                livestock: Sequence[str]) -> None:
    d = data_dir / stage.produces
    if not d.exists():
        return
    prior = read_stamp(stage, data_dir)
    covered = set(livestock)
    # Animals covered accumulate: running dairy today and poultry tomorrow
    # leaves both on disk, so both count as done.
    if prior.get("fingerprint") == fp:
        covered |= set(prior.get("livestock", []))
    d.joinpath(STAMP).write_text(json.dumps({
        "stage": stage.name,
        "fingerprint": fp,
        "livestock": sorted(covered),
    }, indent=2))


def is_current(stage: Stage, cfg: dict, data_dir: Path, scripts_dir: Path,
               livestock: Sequence[str]) -> bool:
    """
    Can this stage be skipped?

    Three conditions, all required: the output exists, it was produced by
    the current script and settings, and -- for a stage that depends on
    the animal -- it already covers every animal being asked for.
    """
    d = data_dir / stage.produces
    if not d.exists() or not any(d.glob("*.feather")):
        return False
    stamp = read_stamp(stage, data_dir)
    if stamp.get("fingerprint") != fingerprint(stage, cfg, scripts_dir):
        return False
    if stage.foundation:
        return True                                # shared, animal-agnostic
    return set(livestock) <= set(stamp.get("livestock", []))


def resolve(chosen: List[Stage], cfg: dict, data_dir: Path,
            scripts_dir: Path, livestock: Sequence[str],
            rebuild: bool) -> tuple[List[Stage], List[Stage]]:
    """
    Work out what actually has to run.

    Two things happen here. Any prerequisite the request depends on but
    does not have is pulled in, recursively -- so asking for the last
    stage on an empty repository builds the whole chain. Then anything
    already produced by the current code, with the current settings, for
    the animals being asked about, is dropped.

    Both are internal. A first-time user runs one command and gets
    everything; a returning user runs the same command and only the parts
    that are missing or out of date are redone. Nothing has to be
    remembered or passed on the command line.
    """
    wanted = {s.name for s in chosen}
    producer = {s.produces: s for s in STAGES}

    pending = list(chosen)
    while pending:
        stage = pending.pop()
        for need in stage.needs:
            dep = producer.get(need)
            if dep is None or dep.name in wanted:
                continue
            wanted.add(dep.name)
            pending.append(dep)

    full = sorted((STAGE_BY_NAME[n] for n in wanted),
                  key=lambda s: ORDER[s.name])

    to_run, already = [], []
    for st in full:
        if not rebuild and is_current(st, cfg, data_dir, scripts_dir,
                                      livestock):
            already.append(st)
        else:
            to_run.append(st)

    # Staleness travels downstream. Rebuilding a stage invalidates
    # everything that consumed its output, even if that output is still
    # sitting on disk looking complete. Without this, editing wcc_mlr.py
    # rebuilds the equations while wcc keeps the county values fitted from
    # the old ones -- the two disagree silently and fail later somewhere
    # unrelated, which is exactly how this pipeline broke once already.
    stale = {st.produces for st in to_run}
    changed = True
    while changed:
        changed = False
        for st in list(already):
            if any(need in stale for need in st.needs):
                already.remove(st)
                to_run.append(st)
                stale.add(st.produces)
                changed = True

    to_run.sort(key=lambda s: ORDER[s.name])
    already.sort(key=lambda s: ORDER[s.name])
    return to_run, already


def check_inputs(stage: Stage, data_dir: Path) -> Optional[str]:
    """Confirm each required input directory exists and is not empty."""
    for need in stage.needs:
        d = data_dir / need
        if not d.exists() or not any(d.glob("*.feather")):
            producer = next((s.name for s in STAGES if s.produces == need),
                            "an earlier stage")
            return (f"{stage.name}: needs data/{need}, which is empty. "
                    f"Run the '{producer}' stage first.")
    return None


def run_stage(stage: Stage, cfg: dict, livestock: Sequence[str],
              scripts_dir: Path, data_dir: Path, dry_run: bool) -> bool:
    argv = to_argv(cfg.get(stage.name))
    label = f"[{stage.group}] {stage.name}"

    if dry_run:
        pl = (f"  livestock: {', '.join(livestock)}"
              if stage.per_livestock and not stage.foundation
              else "  livestock: all (shared foundation)"
              if stage.foundation else "  livestock: n/a")
        print(f"  {label:<24} {stage.module}.py "
              f"{' '.join(argv) if argv else '(no options)'}")
        print(f"  {'':<24} {pl}")
        return True

    problem = check_inputs(stage, data_dir)
    if problem:
        print(f"  SKIPPED  {problem}")
        return False

    # USGS reports one livestock category covering all animals, so a
    # withdrawal comparison built from a subset is measuring the model
    # against a total it was never meant to reproduce. Running dairy alone
    # gives a bias near -80%, which is simply the share of livestock water
    # that is not dairy. Better to decline than to print that number.
    if stage.name == "validate" and set(livestock) != set(LIVESTOCK_ALL):
        missing = sorted(set(LIVESTOCK_ALL) - set(livestock))
        print(f"  SKIPPED  validate needs all four animals; USGS reports "
              f"livestock as a single category, so comparing a subset "
              f"against it is not meaningful.")
        print(f"           missing: {', '.join(missing)}. "
              f"Run without --livestock, or add the rest first.")
        return True

    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    print(f"\n{'=' * 74}")
    print(f"{label}  --  {stage.describe}")
    if (stage.per_livestock and not stage.foundation
            and len(livestock) < len(LIVESTOCK_ALL)):
        print(f"restricted to: {', '.join(livestock)}")
    print("=" * 74)

    try:
        module = importlib.import_module(stage.module)
        importlib.reload(module)          # pick up edits between runs
    except ImportError as err:
        print(f"  FAILED   cannot import {stage.module}: {err}")
        return False

    # A foundation stage is shared, so it is always built for all four
    # animals even when the user asked for one. Narrowing it would leave a
    # half-built foundation that the next request would silently reuse.
    if stage.per_livestock and not stage.foundation:
        narrow_livestock(module, livestock)

    if not hasattr(module, "main"):
        print(f"  FAILED   {stage.module} has no main(); cannot drive it")
        return False

    t0 = time.time()
    try:
        module.main(argv)
    except SystemExit as err:             # a script that still exits
        if err.code not in (0, None):
            print(f"  FAILED   {stage.name} exited with code {err.code}")
            return False
    except Exception:                     # noqa: BLE001 - report and stop
        print(f"  FAILED   {stage.name} raised:")
        traceback.print_exc()
        return False
    write_stamp(stage, data_dir,
                fingerprint(stage, cfg, scripts_dir),
                LIVESTOCK_ALL if stage.foundation else livestock)
    print(f"  done in {time.time() - t0:,.1f}s")
    return True


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def select_stages(args) -> tuple[List[Stage], bool]:
    if args.stage:
        unknown = [s for s in args.stage if s not in STAGE_BY_NAME]
        if unknown:
            raise SystemExit(f"unknown stage(s): {unknown}\n"
                             f"available: {list(STAGE_BY_NAME)}")
        chosen, explicit = [STAGE_BY_NAME[s] for s in args.stage], True
    elif args.from_stage:
        if args.from_stage not in STAGE_BY_NAME:
            raise SystemExit(f"unknown stage: {args.from_stage}")
        start = ORDER[args.from_stage]
        chosen, explicit = [s for s in STAGES if ORDER[s.name] >= start], True
    elif args.group:
        chosen = [s for s in STAGES if s.group.replace(" ", "") ==
                  args.group.replace(" ", "")]
        explicit = True
        if not chosen:
            raise SystemExit(f"unknown group: {args.group}\n"
                             f"available: extract, pathway1, pathway2, evaluate")
    else:
        # A bare `python main.py` is "give me everything", not "rebuild
        # everything": a foundation that is already there is reused.
        chosen, explicit = list(STAGES), False
    # Always dependency order, whatever order they were asked for.
    return sorted(chosen, key=lambda s: ORDER[s.name]), explicit


ANALYSIS = [s.name for s in STAGES if not s.foundation]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run the LivestockWaterUse pipeline.",
        allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="stages: " + ", ".join(s.name for s in STAGES))
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--livestock", nargs="+", default=None,
                    metavar="NAME",
                    help=f"one or more of {', '.join(LIVESTOCK_ALL)}, "
                         f"or 'all'")
    ap.add_argument("--stage", nargs="+", default=None, metavar="NAME",
                    help="run only these stages")
    ap.add_argument("--from", dest="from_stage", default=None, metavar="NAME",
                    help="run this stage and everything after it")
    ap.add_argument("--group", default=None, metavar="NAME",
                    help="extract, pathway1, pathway2 or evaluate")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan without running anything")
    ap.add_argument("--list", action="store_true",
                    help="list the stages and exit")
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild the foundation stages even if their "
                         "output already exists")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="keep going after a stage fails")
    args, unknown = ap.parse_known_args(argv)
    if unknown:
        print(f"(ignoring unrecognized arguments: {unknown})")

    if args.list:
        print(f"{'stage':<10}{'group':<12}{'produces':<38}description")
        for s in STAGES:
            print(f"{s.name:<10}{s.group:<12}"
                  f"{('data/' + s.produces):<38}{s.describe}")
        print("\nfoundation stages are shared by all users and all "
              "animals; they are")
        print("built once, reused afterwards, and always cover all four "
              "livestock types.")
        print("--livestock applies to the analysis stages: "
              + ", ".join(ANALYSIS))
        return 0

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_DEFAULT / cfg_path
    cfg = load_config(cfg_path)

    repo = Path(cfg.get("repo", REPO_DEFAULT)).expanduser()
    scripts_dir, data_dir = repo / "scripts", repo / "data"

    livestock = args.livestock or cfg.get("livestock") or list(LIVESTOCK_ALL)
    if len(livestock) == 1 and livestock[0].lower() == "all":
        livestock = list(LIVESTOCK_ALL)
    bad = [a for a in livestock if a not in LIVESTOCK_ALL]
    if bad:
        raise SystemExit(f"unknown livestock: {bad}\n"
                         f"available: {list(LIVESTOCK_ALL)}, or 'all'")

    stages, _ = select_stages(args)
    stages, already = resolve(stages, cfg, data_dir, scripts_dir,
                              livestock, args.rebuild)

    print("=" * 74)
    print("LivestockWaterUse pipeline")
    print("=" * 74)
    print(f"  config     {cfg_path}")
    print(f"  repo       {repo}")
    print(f"  livestock  {', '.join(livestock)}")
    if already:
        print(f"  up to date {', '.join(s.name for s in already)}")
    print(f"  running    {', '.join(s.name for s in stages) or '(nothing)'}")
    if args.dry_run:
        print("\nplan (nothing will run):")

    # The Earth Engine project has no sensible default and the failure is
    # slow and opaque, so it is checked before anything starts.
    if not stages:
        print("\nnothing to run: everything requested is already built.")
        print("Use --rebuild to redo it.")
        return 0

    if any(s.name == "climate" for s in stages) and not args.dry_run:
        if not (cfg.get("climate") or {}).get("project"):
            raise SystemExit(
                "\nthe 'climate' stage needs an Earth Engine project.\n"
                "Set climate.project in config.yaml, or skip it with "
                "--stage / --from.")

    t0, results = time.time(), []
    for stage in stages:
        ok = run_stage(stage, cfg, livestock, scripts_dir, data_dir,
                       args.dry_run)
        results.append((stage.name, ok))
        if not ok and not args.dry_run and not args.continue_on_error:
            print(f"\nstopped at '{stage.name}'. "
                  f"Fix it, then resume with --from {stage.name}")
            break

    if args.dry_run:
        return 0

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    for name, ok in results:
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}")
    skipped = [s.name for s in stages
               if s.name not in [n for n, _ in results]]
    if skipped:
        print(f"  not reached: {', '.join(skipped)}")
    print(f"\n  total {time.time() - t0:,.1f}s")
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())