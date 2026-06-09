# Model Package File Format

This directory contains information about the `.nnmodel.tar` model package format,
which is used to store trained neural network models with their configuration and
parameters.

## File Format

`.nnmodel.tar` files are **standard POSIX tar archives** containing:

- `model.json` — Model architecture and metadata (JSON format)
- `params.bin` — Trained parameters (weights, biases) in binary format

The `.nnmodel.tar` extension ensures compatibility with standard GUI archive managers
(like GNOME Archive Manager, KDE Ark, Windows Explorer, macOS Archive Utility) on all
platforms without requiring any system-level configuration.

## Opening .nnmodel.tar Files

### Using GUI Archive Managers

**Linux (GNOME/KDE):**
- Double-click the `.nnmodel.tar` file to open it with the default archive manager
- Right-click → Extract Here to extract the contents

**Windows:**
- Double-click the `.nnmodel.tar` file to open it with Windows Explorer
- Right-click → Extract All to extract the contents

**macOS:**
- Double-click the `.nnmodel.tar` file to open it with Archive Utility
- It will extract automatically to the same directory

### Using Command Line

```bash
# List contents
tar -tf model.nnmodel.tar

# Extract to current directory
tar -xf model.nnmodel.tar

# Extract to specific directory
tar -xf model.nnmodel.tar -C /path/to/extract/
```

## Migration from .nnmodel to .nnmodel.tar

If you have existing `.nnmodel` files (older format using only the `.nnmodel` extension),
you can easily rename them:

```bash
# Rename a single file
mv model.nnmodel model.nnmodel.tar

# Rename all .nnmodel files in a directory
for f in *.nnmodel; do mv "$f" "${f%.nnmodel}.nnmodel.tar"; done
```

The NN-CLI, NN-Server, and Python migration script automatically support both
`.nnmodel` and `.nnmodel.tar` extensions, so renamed files will work without
any code changes.

## Creating .nnmodel.tar Files

Use the Python migration script to convert legacy JSON model files:

```bash
python3 scripts/migrate-to-nnmodel.py input.json output.nnmodel.tar
```

If you omit the output path, it will be derived automatically:
```bash
python3 scripts/migrate-to-nnmodel.py input.json
# Creates: input.nnmodel.tar
```

The NN-CLI also creates `.nnmodel.tar` files automatically when saving checkpoints
during training.

## Desktop Integration (Optional, Not Recommended)

The `share/mime/` directory contains MIME type configuration files that could make
`.nnmodel` files (without the `.tar` extension) recognizable as archives on Linux.

**This is no longer recommended** because:
- Requires system-level configuration (sudo or user-specific setup)
- Only works on Linux with GNOME/KDE desktop environments
- Doesn't solve the issue on Windows or macOS
- The `.nnmodel.tar` extension solves the problem more cleanly

If you want to use the desktop integration files anyway, refer to the old
instructions in the source control history. Note that these files are not
installed by the build system and must be installed manually.