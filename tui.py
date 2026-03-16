"""Terminal UI for Song Recommendation System using Textual."""

import os
import sys

import joblib
import pandas as pd
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    LoadingIndicator,
    Static,
)

from src.data_loader import AUDIO_FEATURES, DataLoader
from src.features import FeaturePipeline
from src.recommender import Recommender
from src.utils import (
    DEFAULT_DATASET,
    MODELS_DIR,
    ensure_dirs,
    get_dataset_hash,
    setup_logging,
)

logger = setup_logging()


class LoadingScreen(Screen):
    """Screen shown while models are loading/building."""

    BINDINGS = []

    def compose(self) -> ComposeResult:
        yield Container(
            Vertical(
                Static("🎵 Song Recommendation System", id="loading-title"),
                LoadingIndicator(),
                Static(id="loading-message"),
            ),
            id="loading-container",
        )

    def on_mount(self) -> None:
        self.query_one("#loading-message", Static).update("Initializing...")


class ResultScreen(Screen):
    """Screen to display recommendation results."""

    BINDINGS = [
        Binding("escape", "go_back", "Back", show=True),
        Binding("r", "new_search", "New Search", show=True),
        Binding("up", "scroll_up", "↑", show=True),
        Binding("down", "scroll_down", "↓", show=True),
        Binding("pageup", "page_up", "PgUp", show=True),
        Binding("pagedown", "page_down", "PgDn", show=True),
    ]

    def __init__(self, results: pd.DataFrame, search_type: str, search_query: str):
        super().__init__()
        self.results = results
        self.search_type = search_type
        self.search_query = search_query

    def compose(self) -> ComposeResult:
        yield Header()

        if self.results.empty:
            yield Container(
                Static(
                    f"⚠️  No recommendations found for '{self.search_query}'", id="no-results"
                ),
                id="result-container",
            )
        else:
            yield Container(
                Static(f"🎵 Recommendations for '{self.search_query}'", id="result-title"),
                Static(f"({self.search_type} based)", id="result-subtitle"),
                ScrollableContainer(
                    Static(self._format_results(), id="results-table"),
                    id="results-scroll",
                ),
                id="result-container",
            )

        yield Footer()

    def _format_results(self) -> str:
        """Format results as a table."""
        lines = []
        lines.append(
            "┌─────┬──────────────────────────────────────┬────────────────────────────┬──────────────┐"
        )
        lines.append(
            "│ #   │ Track Name                             │ Artist                     │ Genre        │"
        )
        lines.append(
            "├─────┼──────────────────────────────────────┼────────────────────────────┼──────────────┤"
        )

        for i, (_, row) in enumerate(self.results.head(15).iterrows(), 1):
            track = str(row.get("track_name", ""))[:38]
            artist = str(row.get("artist_name", ""))[:26]
            genre = str(row.get("genre", "unknown"))[:12]
            lines.append(f"│ {i:<3} │ {track:<38} │ {artist:<26} │ {genre:<12} │")

        lines.append(
            "└─────┴──────────────────────────────────────┴────────────────────────────┴──────────────┘"
        )

        if len(self.results) > 15:
            lines.append(f"\n[dim]... and {len(self.results) - 15} more results[/dim]")

        return "\n".join(lines)

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_new_search(self) -> None:
        self.app.pop_screen()
        self.app.push_screen("main_menu")

    def action_scroll_up(self) -> None:
        try:
            self.query_one("#results-scroll", ScrollableContainer).scroll_up()
        except Exception:
            pass

    def action_scroll_down(self) -> None:
        try:
            self.query_one("#results-scroll", ScrollableContainer).scroll_down()
        except Exception:
            pass

    def action_page_up(self) -> None:
        try:
            self.query_one("#results-scroll", ScrollableContainer).page_up()
        except Exception:
            pass

    def action_page_down(self) -> None:
        try:
            self.query_one("#results-scroll", ScrollableContainer).page_down()
        except Exception:
            pass


class MainMenu(Screen):
    """Main menu screen for the TUI."""

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("1", "search_by_song", "By Song", show=True),
        Binding("2", "search_by_artist", "By Artist", show=True),
        Binding("r", "rebuild", "Rebuild Models", show=False),
        Binding("up", "focus_previous", "↑", show=True),
        Binding("down", "focus_next", "↓", show=True),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Vertical(
                Static("🎵 Song Recommendation System", id="menu-title"),
                Static("Choose a search option:", id="menu-subtitle"),
                Button("🎶 Search by Song", id="btn-song", variant="primary"),
                Button("🎤 Search by Artist", id="btn-artist", variant="primary"),
                Button("🔄 Rebuild Models", id="btn-rebuild", variant="default"),
                Button("❌ Exit", id="btn-exit", variant="error"),
            ),
            id="menu-container",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#btn-song", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-song":
            self.app.push_screen("search_song")
        elif event.button.id == "btn-artist":
            self.app.push_screen("search_artist")
        elif event.button.id == "btn-rebuild":
            self.app.rebuild_models()
        elif event.button.id == "btn-exit":
            self.app.exit()

    def action_quit(self) -> None:
        self.app.exit()

    def action_search_by_song(self) -> None:
        self.app.push_screen("search_song")

    def action_search_by_artist(self) -> None:
        self.app.push_screen("search_artist")

    def action_rebuild(self) -> None:
        self.app.rebuild_models()

    def action_focus_previous(self) -> None:
        self.focus_previous()

    def action_focus_next(self) -> None:
        self.focus_next()


class SearchScreen(Screen):
    """Generic search screen with input field."""

    BINDINGS = [
        Binding("escape", "go_back", "Back", show=True),
        Binding("enter", "submit", "Search", show=True),
        Binding("down", "focus_search_button", "Search", show=True),
    ]

    def __init__(self, search_type: str, placeholder: str, title: str):
        super().__init__()
        self.search_type = search_type
        self.placeholder = placeholder
        self.title = title

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Vertical(
                Static(self.title, id="search-title"),
                Input(placeholder=self.placeholder, id="search-input"),
                Horizontal(
                    Button("🔍 Search", id="btn-search", variant="primary"),
                    Button("⬅ Back", id="btn-back", variant="default"),
                    id="search-buttons",
                ),
            ),
            id="search-container",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#search-input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-search":
            self._perform_search()
        elif event.button.id == "btn-back":
            self.app.pop_screen()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._perform_search()

    def _perform_search(self) -> None:
        query = self.query_one("#search-input", Input).value.strip()
        if query:
            self.app.perform_search(query, self.search_type)
        else:
            self.notify("Please enter a search term", severity="warning")

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_submit(self) -> None:
        self._perform_search()

    def action_focus_search_button(self) -> None:
        self.query_one("#btn-search", Button).focus()


class SongRecommenderApp(App):
    """Main TUI application for Song Recommender."""

    # Rose Pine Theme Colors
    # Base: #191724, Surface: #1F1D2E, Overlay: #26233A
    # Muted: #6e6a86, Subtle: #908caa
    # Text: #e0def4, Love: #eb6f92, Gold: #f6c177
    # Rose: #ebbcba, Pine: #31748f, Foam: #9ccfd8

    CSS = """
    /* Rose Pine Theme */
    Screen {
        background: #191724;
    }

    #menu-container, #search-container, #result-container, #loading-container {
        align: center middle;
        height: 100%;
    }

    Vertical {
        align: center middle;
        padding: 2 4;
        background: #1F1D2E;
        border: solid #eb6f92;
        min-width: 60;
    }

    #menu-title {
        text-align: center;
        text-style: bold;
        padding: 1 0 2 0;
        color: #eb6f92;
    }

    #menu-subtitle {
        text-align: center;
        padding: 0 0 2 0;
        color: #908caa;
    }

    #search-title {
        text-align: center;
        text-style: bold;
        padding: 1 0 2 0;
        color: #eb6f92;
    }

    #search-input {
        width: 100%;
        margin: 1 0;
        background: #26233A;
        color: #e0def4;
    }

    #search-buttons {
        width: 100%;
        align: center middle;
        margin-top: 2;
    }

    #search-buttons Button {
        margin: 0 1;
        min-width: 15;
    }

    Button {
        width: 100%;
        margin: 1 0;
        min-width: 30;
        background: #26233A;
        color: #e0def4;
    }

    Button:hover {
        background: #eb6f92;
        color: #191724;
    }

    Button:focus {
        background: #f6c177;
        color: #191724;
    }

    Button#btn-exit {
        background: #524f67;
    }

    #loading-title {
        text-align: center;
        text-style: bold;
        padding: 1 0;
        color: #ebbcba;
    }

    #loading-message {
        text-align: center;
        padding: 1 0;
        color: #908caa;
    }

    #result-title {
        text-align: center;
        text-style: bold;
        padding: 1 0;
        color: #9ccfd8;
    }

    #result-subtitle {
        text-align: center;
        padding: 0 0 1 0;
        color: #908caa;
    }

    #results-table {
        background: #1F1D2E;
        padding: 1 2;
        border: solid #31748f;
        margin-top: 1;
        color: #e0def4;
    }

    #no-results {
        text-align: center;
        color: #f6c177;
        padding: 2;
    }

    Static {
        width: 100%;
    }

    Header {
        background: #1F1D2E;
    }

    Footer {
        background: #1F1D2E;
    }

    LoadingIndicator {
        color: #eb6f92;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, dataset_path: str = None, rebuild: bool = False):
        super().__init__()
        self.dataset_path = dataset_path or str(DEFAULT_DATASET)
        self.force_rebuild = rebuild
        self.loader = None
        self.pipeline = None
        self.recommender = None
        self.df = None
        self.feature_matrix = None
        self.initialized = False

    def on_mount(self) -> None:
        self.push_screen("loading")
        self.initialize_models()

    @work(exclusive=True, thread=True)
    def initialize_models(self) -> None:
        """Initialize models in a background worker."""
        try:
            if not os.path.exists(self.dataset_path):
                self.call_from_thread(
                    self.notify,
                    f"Dataset not found at {self.dataset_path}",
                    severity="error",
                )
                self.call_from_thread(self.exit)
                return

            cached = None if self.force_rebuild else self._load_pipeline()

            if cached:
                self.call_from_thread(
                    self._update_loading_message, "Loaded models from cache ✓"
                )
                (
                    self.loader,
                    self.pipeline,
                    self.recommender,
                    self.df,
                    self.feature_matrix,
                ) = cached
            else:
                self.call_from_thread(
                    self._update_loading_message,
                    "Building models (this may take a while)...",
                )
                (
                    self.loader,
                    self.pipeline,
                    self.recommender,
                    self.df,
                    self.feature_matrix,
                ) = self._build_pipeline()

            self.initialized = True
            self.call_from_thread(self._update_loading_message, "Ready!")

            import time
            time.sleep(0.5)  # Brief pause to show "Ready!"

            self.call_from_thread(self._go_to_main_menu)

        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            self.call_from_thread(self.notify, f"Error: {str(e)}", severity="error")
            self.call_from_thread(self.exit)

    def _update_loading_message(self, message: str) -> None:
        try:
            self.query_one("#loading-message", Static).update(message)
        except Exception:
            pass

    def _go_to_main_menu(self) -> None:
        self.pop_screen()
        self.push_screen("main_menu")

    def _build_pipeline(self):
        """Builds and saves the model pipeline."""
        loader = DataLoader(self.dataset_path)
        df = loader.load_and_preprocess()

        pipeline = FeaturePipeline()
        feature_matrix = pipeline.fit_transform(df, AUDIO_FEATURES)

        recommender = Recommender()
        recommender.fit(feature_matrix)

        ensure_dirs()
        pipeline.save_artifacts()
        recommender.save_model()

        joblib.dump(df, MODELS_DIR / "processed_df.pkl")
        ds_hash = get_dataset_hash(self.dataset_path)
        with open(MODELS_DIR / "dataset_hash.txt", "w") as f:
            f.write(ds_hash)

        joblib.dump(feature_matrix, MODELS_DIR / "feature_matrix.pkl")

        return loader, pipeline, recommender, df, feature_matrix

    def _load_pipeline(self):
        """Loads existing model pipeline if hashes match."""
        hash_path = MODELS_DIR / "dataset_hash.txt"
        if not hash_path.exists():
            return None

        with open(hash_path, "r") as f:
            saved_hash = f.read().strip()

        if saved_hash != get_dataset_hash(self.dataset_path):
            logger.info("Dataset changed. Rebuilding pipeline...")
            return None

        try:
            df = joblib.load(MODELS_DIR / "processed_df.pkl")
            pipeline = FeaturePipeline()
            pipeline.load_artifacts()

            recommender = Recommender()
            recommender.load_model()

            matrix_path = MODELS_DIR / "feature_matrix.pkl"
            if matrix_path.exists():
                feature_matrix = joblib.load(matrix_path)
            else:
                feature_matrix = pipeline.fit_transform(df, AUDIO_FEATURES)
                joblib.dump(feature_matrix, matrix_path)

            loader = DataLoader(self.dataset_path)
            loader.df = df
            loader._build_indices()

            return loader, pipeline, recommender, df, feature_matrix
        except Exception as e:
            logger.error(f"Error loading pipeline: {e}")
            return None

    @work(exclusive=True, thread=True)
    def perform_search(self, query: str, search_type: str):
        """Perform search in background."""
        try:
            if not self.initialized:
                self.call_from_thread(
                    self.notify, "Models not ready yet", severity="warning"
                )
                return

            if search_type == "song":
                results = self.recommender.get_recommendations(
                    query, self.df, self.feature_matrix
                )
            else:
                results = self.recommender.get_recommendations_by_artist(
                    query, self.df, self.feature_matrix
                )

            self.call_from_thread(
                self._show_results,
                results,
                "Song" if search_type == "song" else "Artist",
                query,
            )
        except Exception as e:
            logger.error(f"Search error: {e}")
            self.call_from_thread(self.notify, f"Error: {str(e)}", severity="error")

    def _show_results(self, results: pd.DataFrame, search_type: str, query: str):
        self.push_screen(ResultScreen(results, search_type, query))

    @work(exclusive=True, thread=True)
    def rebuild_models(self):
        """Rebuild all models."""
        try:
            self.notify("Rebuilding models...", timeout=2)
            self.force_rebuild = True
            self.pop_screen()
            self.push_screen("loading")
            self.initialize_models()
        except Exception as e:
            self.notify(f"Error rebuilding: {e}", severity="error")

    def action_quit(self) -> None:
        self.exit()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Terminal UI Song Recommender")
    parser.add_argument(
        "--dataset", type=str, default=str(DEFAULT_DATASET), help="Path to CSV dataset"
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="Force rebuild of models"
    )
    args = parser.parse_args()

    app = SongRecommenderApp(dataset_path=args.dataset, rebuild=args.rebuild)

    # Register screens
    app.install_screen(MainMenu, name="main_menu")
    app.install_screen(
        SearchScreen(
            search_type="song",
            placeholder="Enter song name...",
            title="🎶 Search by Song",
        ),
        name="search_song",
    )
    app.install_screen(
        SearchScreen(
            search_type="artist",
            placeholder="Enter artist name...",
            title="🎤 Search by Artist",
        ),
        name="search_artist",
    )
    app.install_screen(LoadingScreen, name="loading")

    app.run()


if __name__ == "__main__":
    main()
