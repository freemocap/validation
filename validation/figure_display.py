import os


SHOW_FIGURES_ENV = "VALIDATION_SHOW_FIGURES"


def should_show_figures() -> bool:
    value = os.getenv(SHOW_FIGURES_ENV)

    if value is None:
        return True

    return value.lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def show_figure(fig) -> None:
    if should_show_figures():
        fig.show()