
from pathlib import Path

# Supported image formats
SUPPORTED_FORMATS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
    ".heic",
    ".heif"
}


def scan_images(folder_path: str, recursive: bool = True) -> list[Path]:
    """
    Scan a directory and return all supported image files.

    Args:
        folder_path: Directory to scan.
        recursive: Whether to scan subdirectories.

    Returns:
        A list of image file paths.
    """

    folder = Path(folder_path)

    # Validate directory
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    # Find image files
    if recursive:
        files = folder.rglob("*")
    else:
        files = folder.iterdir()

    images = [
        file for file in files
        if file.is_file()
        and file.suffix.lower() in SUPPORTED_FORMATS
    ]

    return sorted(images)


if __name__ == "__main__":

    folder = input("Enter the folder path containing images: ")

    try:
        images = scan_images(folder)

        print("\nImage scanning completed!")
        print(f"Total images found: {len(images)}")

        for image in images:
            print(image)

    except (FileNotFoundError, NotADirectoryError) as error:
        print(f"Error: {error}")
