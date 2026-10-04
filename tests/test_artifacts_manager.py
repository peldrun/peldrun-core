"""
Automated Verification Suite for Phase 9 Artifacts Subsystem.
Verifies baseline delta scanning, artifact classification, and manifest compilation.
"""

import os
import shutil
import tempfile
import pytest

from peldrun.artifacts.manager import ArtifactManager
from peldrun.artifacts.manifest import ArtifactType


@pytest.mark.asyncio
async def test_artifacts_manager_delta_scanning():
    temp_dir = tempfile.mkdtemp()
    try:
        manager = ArtifactManager(workspace_root=temp_dir)

        # 1. Create a pre-existing baseline file
        baseline_file = os.path.join(temp_dir, "existing.txt")
        with open(baseline_file, "w", encoding="utf-8") as f:
            f.write("I existed before the run.")

        manager.snapshot_baseline()

        # 2. Simulate agent creating deliverables
        code_file = os.path.join(temp_dir, "solution.py")
        with open(code_file, "w", encoding="utf-8") as f:
            f.write("print('Hello PELDRUN')")

        doc_file = os.path.join(temp_dir, "report.md")
        with open(doc_file, "w", encoding="utf-8") as f:
            f.write("# Verification Report\nDone.")

        # 3. Scan for new artifacts
        new_artifacts = await manager.scan_new_artifacts(compute_hashes=True)

        assert len(new_artifacts) == 2
        names = {art.name for art in new_artifacts}
        assert names == {"solution.py", "report.md"}

        # 4. Verify classifications
        code_art = manager.get_manifest().get_by_name("solution.py")
        assert code_art is not None
        assert code_art.artifact_type == ArtifactType.CODE
        assert code_art.sha256 is not None

        doc_art = manager.get_manifest().get_by_name("report.md")
        assert doc_art is not None
        assert doc_art.artifact_type == ArtifactType.DOCUMENT

        # 5. Summary formatting
        summary = manager.format_deliverables_summary()
        assert "solution.py" in summary
        assert "report.md" in summary

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)