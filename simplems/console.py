from rich.console import Console
from rich.theme import Theme

color_theme = Theme({
    "black": "black",
    "red": "red",
    "green": "green",
    "yellow": "yellow",
    "blue": "blue",
    "magenta": "magenta",
    "cyan": "cyan",
    "white": "white",

    "bright_red": "bright_red",
    "bright_green": "bright_green",
    "bright_yellow": "bright_yellow",
    "bright_blue": "bright_blue",
    "bright_magenta": "bright_magenta",
    "bright_cyan": "bright_cyan",

    "class_name": "yellow",
    "warning": "magenta",
    "field": "cyan",
    "value": "green",
})

console = Console(theme=color_theme)

