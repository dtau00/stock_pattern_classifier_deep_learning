"""
Metadata manager for tracking downloaded data packages.
"""
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path


class MetadataManager:
    """
    Manages metadata for downloaded data packages.
    Stores package information in JSON format.
    """

    def __init__(self, metadata_file: str = "data/metadata/packages.json"):
        """
        Initialize metadata manager.

        Args:
            metadata_file: Path to metadata JSON file
        """
        self.metadata_file = Path(metadata_file)
        self._ensure_metadata_file()

    def _ensure_metadata_file(self):
        """Create metadata file if it doesn't exist."""
        # Create directory if needed
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)

        # Create empty metadata file if it doesn't exist
        if not self.metadata_file.exists():
            self._save_metadata([])

    def _load_metadata(self) -> List[Dict]:
        """
        Load metadata from JSON file.

        Returns:
            List of package metadata dictionaries
        """
        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save_metadata(self, metadata: List[Dict]):
        """
        Save metadata to JSON file.

        Args:
            metadata: List of package metadata dictionaries
        """
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def add_package(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str,
        total_bars: int,
        file_path: str,
        file_size: int,
        validation_report: Optional[Dict] = None
    ) -> str:
        """
        Add a new package to metadata.

        Args:
            symbol: Trading pair symbol
            interval: Kline interval
            start_date: Start date string
            end_date: End date string
            total_bars: Number of bars in package
            file_path: Path to data file
            file_size: File size in bytes
            validation_report: Optional validation report from data fetcher

        Returns:
            Package ID (unique identifier)
        """
        metadata = self._load_metadata()

        # Generate package ID
        package_id = f"{symbol}_{interval}_{start_date}_{end_date}".replace(':', '-').replace(' ', '_')

        # Create package entry
        package = {
            'package_id': package_id,
            'symbol': symbol,
            'interval': interval,
            'start_date': start_date,
            'end_date': end_date,
            'total_bars': total_bars,
            'file_path': file_path,
            'file_size': file_size,
            'file_size_mb': round(file_size / (1024 * 1024), 2),
            'download_timestamp': datetime.now().isoformat(),
            'validation_report': validation_report or {}
        }

        # Check if package already exists (update instead of duplicate)
        existing_idx = None
        for i, pkg in enumerate(metadata):
            if pkg['package_id'] == package_id:
                existing_idx = i
                break

        if existing_idx is not None:
            # Update existing package
            metadata[existing_idx] = package
        else:
            # Add new package
            metadata.append(package)

        self._save_metadata(metadata)
        return package_id

    def get_package(self, package_id: str) -> Optional[Dict]:
        """
        Get package metadata by ID.

        Args:
            package_id: Package identifier

        Returns:
            Package metadata dictionary or None if not found
        """
        metadata = self._load_metadata()
        for package in metadata:
            if package['package_id'] == package_id:
                return package
        return None

    def get_all_packages(self) -> List[Dict]:
        """
        Get all package metadata.

        Returns:
            List of all package metadata dictionaries
        """
        return self._load_metadata()

    def delete_package(self, package_id: str) -> bool:
        """
        Delete package metadata and associated file.

        Args:
            package_id: Package identifier

        Returns:
            True if deleted successfully, False if package not found
        """
        metadata = self._load_metadata()

        # Find and remove package
        package_idx = None
        package_data = None
        for i, pkg in enumerate(metadata):
            if pkg['package_id'] == package_id:
                package_idx = i
                package_data = pkg
                break

        if package_idx is None:
            return False

        # Delete file if it exists
        if package_data and 'file_path' in package_data:
            file_path = Path(package_data['file_path'])
            if file_path.exists():
                try:
                    file_path.unlink()
                except Exception as e:
                    print(f"Warning: Could not delete file {file_path}: {e}")

        # Remove from metadata
        metadata.pop(package_idx)
        self._save_metadata(metadata)

        return True

    def get_package_count(self) -> int:
        """
        Get total number of packages.

        Returns:
            Number of packages
        """
        return len(self._load_metadata())

    def get_total_storage_size(self) -> int:
        """
        Get total storage size of all packages.

        Returns:
            Total size in bytes
        """
        metadata = self._load_metadata()
        return sum(pkg.get('file_size', 0) for pkg in metadata)

    def search_packages(
        self,
        symbol: Optional[str] = None,
        interval: Optional[str] = None
    ) -> List[Dict]:
        """
        Search packages by symbol and/or interval.

        Args:
            symbol: Filter by symbol (optional)
            interval: Filter by interval (optional)

        Returns:
            List of matching packages
        """
        metadata = self._load_metadata()
        results = []

        for package in metadata:
            if symbol and package['symbol'] != symbol:
                continue
            if interval and package['interval'] != interval:
                continue
            results.append(package)

        return results


if __name__ == "__main__":
    import sys
    import io

    # Fix Unicode output on Windows
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 60)
    print("Testing MetadataManager")
    print("=" * 60)

    # Create test metadata manager
    test_file = "data/metadata/test_packages.json"
    manager = MetadataManager(test_file)

    # Test 1: Add package
    print("\n[Test 1] Adding package...")
    package_id = manager.add_package(
        symbol='BTCUSDT',
        interval='1h',
        start_date='2024-01-01',
        end_date='2024-01-10',
        total_bars=217,
        file_path='data/packages/test.csv',
        file_size=50000,
        validation_report={'data_quality': 'good', 'gap_count': 0}
    )
    print(f"  Added package: {package_id}")
    assert package_id == "BTCUSDT_1h_2024-01-01_2024-01-10"
    print("  [OK]")

    # Test 2: Get package
    print("\n[Test 2] Retrieving package...")
    package = manager.get_package(package_id)
    assert package is not None
    assert package['symbol'] == 'BTCUSDT'
    assert package['total_bars'] == 217
    print(f"  Retrieved: {package['symbol']} {package['interval']}")
    print(f"  Bars: {package['total_bars']}")
    print(f"  Size: {package['file_size_mb']} MB")
    print("  [OK]")

    # Test 3: Add another package
    print("\n[Test 3] Adding second package...")
    package_id2 = manager.add_package(
        symbol='ETHUSDT',
        interval='1h',
        start_date='2024-01-01',
        end_date='2024-01-10',
        total_bars=217,
        file_path='data/packages/test2.csv',
        file_size=48000
    )
    print(f"  Added package: {package_id2}")
    print("  [OK]")

    # Test 4: Get all packages
    print("\n[Test 4] Getting all packages...")
    all_packages = manager.get_all_packages()
    assert len(all_packages) == 2
    print(f"  Found {len(all_packages)} packages")
    for pkg in all_packages:
        print(f"    - {pkg['symbol']} {pkg['interval']}")
    print("  [OK]")

    # Test 5: Search packages
    print("\n[Test 5] Searching packages...")
    btc_packages = manager.search_packages(symbol='BTCUSDT')
    assert len(btc_packages) == 1
    print(f"  Found {len(btc_packages)} BTCUSDT packages")
    print("  [OK]")

    # Test 6: Get package count
    print("\n[Test 6] Package count...")
    count = manager.get_package_count()
    assert count == 2
    print(f"  Total packages: {count}")
    print("  [OK]")

    # Test 7: Get total storage
    print("\n[Test 7] Total storage size...")
    total_size = manager.get_total_storage_size()
    print(f"  Total size: {total_size} bytes ({total_size / 1024:.2f} KB)")
    print("  [OK]")

    # Test 8: Update existing package
    print("\n[Test 8] Updating existing package...")
    package_id = manager.add_package(
        symbol='BTCUSDT',
        interval='1h',
        start_date='2024-01-01',
        end_date='2024-01-10',
        total_bars=220,  # Updated
        file_path='data/packages/test_updated.csv',
        file_size=51000
    )
    updated = manager.get_package(package_id)
    assert updated['total_bars'] == 220
    print(f"  Updated bars: {updated['total_bars']}")
    print("  [OK]")

    # Test 9: Delete package
    print("\n[Test 9] Deleting package...")
    deleted = manager.delete_package(package_id2)
    assert deleted is True
    count_after = manager.get_package_count()
    assert count_after == 1
    print(f"  Packages after delete: {count_after}")
    print("  [OK]")

    # Clean up test file
    print("\n[Cleanup] Removing test metadata file...")
    Path(test_file).unlink()
    print("  [OK]")

    print("\n" + "=" * 60)
    print("[OK] All MetadataManager tests passed!")
    print("=" * 60)
