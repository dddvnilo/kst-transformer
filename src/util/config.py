"""
Ucitavanje argumenata skripti iz TOML config fajla.

Vrednosti se slazu u tri sloja, od najslabijeg ka najjacem:
    default iz parsera  <  config fajl  <  komandna linija

Kljucevi u config fajlu su imena argumenata sa donjom crtom umesto crtice,
npr. `closure_loss = true` za --closure-loss.
"""

import argparse
import tomllib


def parse_args_with_config(parser: argparse.ArgumentParser, argv=None) -> argparse.Namespace:
    parser.add_argument("--config", type=str, default=None,
                        help="TOML fajl sa vrednostima argumenata; argumenti sa komandne linije imaju prednost")
    known, _ = parser.parse_known_args(argv)

    if known.config is not None:
        with open(known.config, "rb") as f:
            config = tomllib.load(f)

        actions = {a.dest: a for a in parser._actions if a.dest not in ("help", "config")}
        unknown = sorted(set(config) - set(actions))
        if unknown:
            parser.error(f"Nepoznati kljucevi u {known.config}: {unknown}")
        # argparse ne proverava `choices` za default vrednosti, pa to radimo rucno
        for key, value in config.items():
            choices = actions[key].choices
            if choices is not None and value not in choices:
                parser.error(f"{known.config}: {key} = {value!r} nije jedno od {list(choices)}")

        parser.set_defaults(**config)

    return parser.parse_args(argv)
