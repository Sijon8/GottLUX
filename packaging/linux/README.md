# Linux desktop integration

Freedesktop (XDG) integration files for GottLUX:

| file                   | what it does                                                        |
| ---------------------- | ------------------------------------------------------------------- |
| `gottlux-view.desktop` | desktop entry for the quick viewer (`gottlux-view`), the `.raw` handler |
| `gottlux-gui.desktop`  | desktop entry for the full tabbed dashboard (`gottlux-gui`)         |
| `gottlux-raw.xml`      | shared-mime-info package declaring `application/x-gottlux-raw` for `*.raw` (low glob weight — `.raw` also matches camera photo formats) |

## Automatic install (recommended)

```sh
gottlux-view --register      # or: ./gottlux_view.sh --register from a source checkout
```

writes the mime + desktop files into `~/.local/share/`, refreshes the mime/desktop
databases, and sets the default handler. `gottlux-view --unregister` undoes it.
(`gottlux-view.desktop` and `gottlux-raw.xml` here are the same content `--register`
generates — `gottlux.app.file_assoc` is the single source of truth.)

## Manual install (per-user)

```sh
install -Dm644 gottlux-raw.xml      ~/.local/share/mime/packages/gottlux-raw.xml
install -Dm644 gottlux-view.desktop ~/.local/share/applications/gottlux-view.desktop
install -Dm644 gottlux-gui.desktop  ~/.local/share/applications/gottlux-gui.desktop
update-mime-database ~/.local/share/mime
update-desktop-database ~/.local/share/applications
xdg-mime default gottlux-view.desktop application/x-gottlux-raw
```

For a system-wide install use `/usr/share/` instead of `~/.local/share/` (and run the
database updates against those directories). The `Exec=` lines assume the `gottlux-view` /
`gottlux-gui` console scripts are on `PATH` (i.e. the package is `pip install`ed); for a
source checkout, `--register` writes an explicit `python -m gottlux.app.quickview` command
instead.
