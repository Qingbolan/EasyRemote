from pathlib import Path
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.box import ROUNDED
from rich.theme import Theme
import logging
import os
from typing import Any, Optional, Tuple
from pyfiglet import Figlet

class CompactRichHandler(RichHandler):
    """
    Custom RichHandler that prevents extra spacing between log messages.
    
    This handler overrides the default RichHandler behavior to provide
    compact logging output without the extra blank lines that Rich
    sometimes adds between log messages.
    """
    
    def __init__(self, *args, log_time_format="%H:%M:%S", **kwargs):
        super().__init__(*args, **kwargs)
        # Store the time format
        self.log_time_format = log_time_format
    
    def emit(self, record):
        """
        Override emit to control spacing and format.
        """
        try:
            # Get the message text
            message = record.getMessage()
            
            # Format time using the formatter
            import time
            time_str = time.strftime(self.log_time_format, time.localtime(record.created))
            
            # Format level with style
            level_name = record.levelname
            level_style = self._get_level_style(level_name)
            
            # Create compact log line with Rich markup
            log_line = f"[dim]{time_str}[/dim] [{level_style}]{level_name:<8}[/{level_style}] {message}"
            
            # Print directly without Rich's extra formatting
            self.console.print(log_line, highlight=False, markup=True)
            
        except Exception:
            self.handleError(record)
    
    def _get_level_style(self, level_name):
        """Get style for log level."""
        styles = {
            "DEBUG": "grey70",
            "INFO": "bold cyan", 
            "WARNING": "bold yellow",
            "ERROR": "bold red",
            "CRITICAL": "bold white on red"
        }
        return styles.get(level_name, "bold")

class ModernLogger:
    """
    Modern colorful logger with smooth gradient styling inspired by Vue CLI,
    built on top of the rich library.
    
    Features:
    - Compact mode to eliminate extra spacing between log messages
    - Rich formatting with emoji icons and color themes
    - Gradient text effects for special messages
    - Progress bars, panels, and other rich UI elements
    
    Args:
        name: Logger name (default: "app")
        level: Log level ("debug", "info", "warning", "error", "critical")
        log_file: Optional file path for logging to file
        show_path: Whether to show file paths in logs
        rich_tracebacks: Enable rich traceback formatting
        compact_mode: If True, eliminates extra spacing between log messages (default: True)
    """

    # Emoji prefixes for each level
    LEVEL_ICONS = {
        "DEBUG": "🐛",
        "INFO": "ℹ️",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "CRITICAL": "💀"
    }

    # Hardcoded gradient endpoints
    GRADIENT_START = "#41B883"  # Vue green
    GRADIENT_END   = "#6574CD"  # Vue purple

    def __init__(
        self,
        name: str = "app",
        level: str = "info",
        log_file: Optional[str] = None,
        show_path: bool = True,
        rich_tracebacks: bool = True,
        compact_mode: bool = True
    ):
        # Enable rich tracebacks
        install_rich_traceback(show_locals=True)

        # Map text levels to logging levels
        levels = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL
        }
        log_level = levels.get(level.lower(), logging.INFO)

        # Define custom theme (directly using color/effect strings)
        custom_theme = Theme({
            "info": "bold cyan",
            "warning": "bold yellow",
            "error": "bold red",
            "critical": "bold white on red",
            "success": "#4CAF50",
            "vue_primary": "#42B883",
            "vue_secondary": "#35495E",
            "logging.time": "dim white",
            "logging.level.debug": "grey70",
            "logging.level.info": "bold cyan",
            "logging.level.warning": "bold yellow",
            "logging.level.error": "bold red",
            "logging.level.critical": "bold white on red",
        })

        # Store compact mode setting
        self.compact_mode = compact_mode
        
        # Set up console and handler
        self.console = Console(theme=custom_theme, highlight=True)
        
        if compact_mode:
            # Use custom compact handler
            rich_handler = CompactRichHandler(
                console=self.console,
                show_time=True,
                show_level=True,
                show_path=False,
                markup=True,
                rich_tracebacks=False,
                log_time_format="%H:%M:%S"
            )
        else:
            # Use standard RichHandler
            rich_handler = RichHandler(
                console=self.console,
                show_time=True,
                show_level=True,
                show_path=False,  # Don't show file paths to reduce clutter
                markup=True,
                rich_tracebacks=False,  # Disable rich tracebacks to avoid multiline
                log_time_format="%H:%M:%S",
                omit_repeated_times=False,  # Prevent extra spacing
                keywords=[]  # Disable keyword highlighting to reduce spacing
            )

        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.logger.handlers.clear()
        self.logger.addHandler(rich_handler)

        if log_file:
            self._setup_file_handler(log_file)

    def _setup_file_handler(self, log_file: str) -> None:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        self.logger.addHandler(fh)

    def _create_gradient_text(self, text: str) -> Text:
        """
        Create a smooth gradient across the text from GRADIENT_START to GRADIENT_END.
        """
        def hex_to_rgb(h: str) -> Tuple[int,int,int]:
            return tuple(int(h[i:i+2], 16) for i in (1, 3, 5))

        start_rgb = hex_to_rgb(self.GRADIENT_START)
        end_rgb   = hex_to_rgb(self.GRADIENT_END)
        length = len(text)
        grad_text = Text()

        for i, ch in enumerate(text):
            t = i / (length - 1) if length > 1 else 0
            r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * t)
            g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * t)
            b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * t)
            color = f"#{r:02X}{g:02X}{b:02X}"
            grad_text.append(ch, style=color)

        return grad_text

    def _print_with_spacing(self, *args, **kwargs) -> None:
        """
        Print with optional spacing based on compact mode.
        """
        if not self.compact_mode:
            self.console.print()
        self.console.print(*args, **kwargs)
        if not self.compact_mode:
            self.console.print()

    # —— Logging Level Methods —— #

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.logger.debug(f"{self.LEVEL_ICONS['DEBUG']} {message}", *args, **kwargs)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.logger.info(f"{self.LEVEL_ICONS['INFO']} {message}", *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.logger.warning(f"{self.LEVEL_ICONS['WARNING']} {message}", *args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.logger.error(f"{self.LEVEL_ICONS['ERROR']} {message}", *args, **kwargs)

    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.logger.critical(f"{self.LEVEL_ICONS['CRITICAL']} {message}", *args, **kwargs)

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.logger.exception(f"{self.LEVEL_ICONS['ERROR']} {message}", *args, **kwargs)

    # —— Rich Print Methods —— #

    def print(self, *args: Any, **kwargs: Any) -> None:
        self.console.print(*args, **kwargs)

    def progress(self, total: int = 100, description: str = "Processing") -> Tuple[Progress, int]:
        """
        Create a lively progress bar with smooth gradient fill.
        Shows elapsed time, estimated time remaining and completion status.
        """
        progress = Progress(
            SpinnerColumn("bouncingBall", style="vue_secondary"),
            TextColumn("[bold vue_primary]{task.description}"),
            BarColumn(complete_style=self.GRADIENT_START, finished_style=self.GRADIENT_END),
            TaskProgressColumn("[bold vue_secondary]{task.percentage:>3.0f}%"),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            expand=True
        )
        task_id = progress.add_task(description, total=total)
        return progress, task_id

    def stage(self, message: str) -> None:
        """
        Display a prominent stage marker with smooth gradient panel.
        """
        panel = Panel(
            self._create_gradient_text(f" {message} "),
            box=ROUNDED,
            border_style="vue_primary",
            expand=False,
            padding=(0, 2)
        )
        if self.compact_mode:
            self.console.print(panel, justify="left")
        else:
            self._print_with_spacing(panel, justify="left")

    def highlight(self, message: str) -> None:
        """
        Print a star marker followed by gradient text.
        """
        txt = Text(" ★ ", style=f"bold {self.GRADIENT_START}")
        txt.append(self._create_gradient_text(message))
        self.console.print(txt)

    def success(self, message: str) -> None:
        """
        Print a green checkmark with message.
        """
        txt = Text()
        txt.append(" ✓ ", style="bold green")
        txt.append(message, style="green")
        self.console.print(txt)

    def error_box(self, message: str) -> None:
        panel = Panel(
            message,
            title="ERROR",
            title_align="left",
            border_style="bold red",
            box=ROUNDED,
            padding=(1, 2)
        )
        self.console.print(panel)

    def section(self, message: str) -> None:
        """
        Print a gradient-colored rule with section title.
        """
        header = self._create_gradient_text(f" {message} ")
        if self.compact_mode:
            self.console.rule(header, style="vue_secondary", align="center")
        else:
            if not self.compact_mode:
                self.console.print()
            self.console.rule(header, style="vue_secondary", align="center")
            if not self.compact_mode:
                self.console.print()

    def info_panel(self, title: str, message: str) -> None:
        panel = Panel(
            message,
            title=title,
            title_align="left",
            border_style="vue_primary",
            box=ROUNDED,
            padding=(1, 2)
        )
        self.console.print(panel)

    def gradient_text(self, message: str) -> None:
        """
        Print a standalone smooth-gradient text.
        """
        self.console.print(self._create_gradient_text(message))

    def table(self, title: Optional[str] = None) -> Table:
        tbl = Table(
            title=title,
            box=ROUNDED,
            title_style="bold vue_primary",
            border_style="vue_primary",
            header_style="bold"
        )
        return tbl

    # —— New: File Save Notification —— #

    def file_saved(self, file_path: str, file_name: str=None) -> None:
        """
        Print "File saved" with path as clickable link and emoji indicator.
        """
        uri = Path(file_path).as_uri()
        # Use Rich Markup to generate linked text
        if file_name:
            self.console.print(
                f"💾 ({file_name}) File saved: [link={uri}][bold blue]{file_path}[/bold blue][/link]"
            )
        else:
            self.console.print(
                f"💾 File saved: [link={uri}][bold blue]{file_path}[/bold blue][/link]"
            )
    
        
    def banner(self, project_name: str, title: str, description: str, font: str = "slant") -> None:
        """
        Print an ASCII Art Banner at program startup with gradient styling,
        and a project description panel using the theme colors.
        """
        fig = Figlet(font=font)
        art = fig.renderText(project_name)

        # Apply character-level gradient to each line
        for line in art.splitlines():
            grad_line = self._create_gradient_text(line)
            self.console.print(grad_line)

        panel = Panel(
            description,
            title=f"[bold vue_primary]{title}",
            border_style="vue_primary",
            box=ROUNDED,
            padding=(1, 2),
        )
        self.console.print(panel)


