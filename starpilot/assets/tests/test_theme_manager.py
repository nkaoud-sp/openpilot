from openpilot.starpilot.assets import theme_manager
from openpilot.starpilot.assets.theme_manager import ThemeManager


def test_update_themes_falls_back_when_first_source_has_no_assets(monkeypatch):
  manager = object.__new__(ThemeManager)
  manager.downloading_theme = False
  manager.session = object()

  sources = ["https://huggingface.example/resources", "https://github.example/resources"]
  fetched_sources = []
  captured = {}

  monkeypatch.setattr(theme_manager, "get_resource_urls", lambda session: sources)
  monkeypatch.setattr(ThemeManager, "sync_local_resources", lambda self: None)

  def fake_fetch_assets(self, repo_url, starpilot_toggles):
    fetched_sources.append(repo_url)
    if repo_url == sources[0]:
      return {"boot_logos": [], "themes": {}, "wheels": []}
    return {"boot_logos": ["bootlogo/FrogPilotModern.png"], "themes": {}, "wheels": []}

  def fake_update_theme_params(self, boot_logos, colors, distance_icons, icons, signals, sounds, wheels):
    captured["boot_logos"] = boot_logos
    captured["colors"] = colors
    captured["distance_icons"] = distance_icons
    captured["icons"] = icons
    captured["signals"] = signals
    captured["sounds"] = sounds
    captured["wheels"] = wheels

  monkeypatch.setattr(ThemeManager, "fetch_assets", fake_fetch_assets)
  monkeypatch.setattr(ThemeManager, "update_theme_params", fake_update_theme_params)

  manager.update_themes(starpilot_toggles=object())

  assert fetched_sources == sources
  assert captured == {
    "boot_logos": ["FrogPilotModern"],
    "colors": [],
    "distance_icons": [],
    "icons": [],
    "signals": [],
    "sounds": [],
    "wheels": [],
  }


def test_has_downloadable_assets_requires_non_empty_asset_groups():
  assert not ThemeManager.has_downloadable_assets({})
  assert not ThemeManager.has_downloadable_assets({"boot_logos": [], "themes": {}, "wheels": []})
  assert ThemeManager.has_downloadable_assets({"boot_logos": ["Stock.png"], "themes": {}, "wheels": []})
