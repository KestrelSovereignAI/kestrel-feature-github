"""GitHub Feature - Repository access and code introspection."""
import logging
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Optional

import yaml

from kestrel_sdk.features.base import Feature, tool
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult

from .ast_analyzer import ASTAnalyzer
from .cache import GitHubCache
from .client import GitHubClient, GitHubClientError
from .models import ComponentManifest, FileType
from .stale_work import classify_stale_work
from .stalled_sweep_source import (
    SOURCE_NAME as STALLED_SWEEP_SOURCE_NAME,
    build_fleet_stalled_sweep_registration,
)

logger = logging.getLogger(__name__)


# Configuration
GITHUB_SELF_REPO = os.getenv("GITHUB_SELF_REPO", "KestrelSovereignAI/kestrel-sovereign")
GITHUB_DEFAULT_BRANCH = os.getenv("GITHUB_DEFAULT_BRANCH", "main")
GITHUB_SELF_FEATURES_ROOT = os.getenv("GITHUB_SELF_FEATURES_ROOT", "kestrel_sovereign/features")


def _configured_fleet_repos() -> list[str]:
    """Operator-configured fleet the agent watches (comma-separated env).

    The fleet membership is a deployment fact, not something to hardcode, so it
    lives in config next to ``GITHUB_SELF_REPO``. Empty/unset => just the
    agent's own repo, so ``repos="fleet"`` always resolves to something useful.
    """
    raw = os.getenv("GITHUB_FLEET_REPOS", "")
    repos = [r.strip() for r in raw.split(",") if r.strip()]
    return repos or [GITHUB_SELF_REPO]


class GitHubFeature(Feature):
    """Feature for accessing GitHub repositories and code introspection.

    Supports:
    - Reading files from any GitHub repository
    - Listing directory contents
    - Searching code
    - AST-based code analysis for Python files
    - Component manifest discovery for self-introspection
    - "self" alias for the agent's own codebase
    """

    tool_name = "github"
    tool_description = "Access GitHub repositories, read source code, and analyze the agent's own codebase"

    def __init__(self, agent=None):
        """Initialize GitHub feature."""
        super().__init__(agent)
        self._client: Optional[GitHubClient] = None
        self._cache: Optional[GitHubCache] = None

    async def initialize(self):
        """Initialize the feature."""
        # Client and cache are lazily initialized
        self._register_stalled_sweep_source()

    def _register_stalled_sweep_source(self) -> None:
        """Register the fleet_stalled_sweep ACTION source with the agent's
        signal registry, if one is available (the rescue workflow's detect
        stage dispatches to it). Idempotent and best-effort."""
        signal_registry = getattr(self.agent, "signal_registry", None)
        if signal_registry is None:
            return
        try:
            if signal_registry.get(STALLED_SWEEP_SOURCE_NAME) is None:
                signal_registry.register(
                    build_fleet_stalled_sweep_registration(self._fleet_sweep_handler)
                )
        except Exception as exc:  # noqa: BLE001 - registration is best-effort
            logger.warning(
                "Could not register fleet_stalled_sweep signal source: %s", exc
            )

    async def _fleet_sweep_handler(self, payload: dict) -> dict:
        """ACTION handler for fleet_stalled_sweep: scan the configured fleet
        (or an explicit repos spec) and return structured findings."""
        stale_days = int(payload.get("stale_days", 3))
        slugs = self._resolve_repos(str(payload.get("repos", "fleet")))
        findings, errors = await self._scan_repos(slugs, stale_days)
        return {
            "findings": findings,
            "errors": errors,
            "repos_scanned": slugs,
            "stale_days": stale_days,
        }

    @property
    def is_available(self) -> bool:
        """Check if the GitHub feature is available (has token configured)."""
        return self.client._configured

    @property
    def client(self) -> GitHubClient:
        """Get or create GitHub client."""
        if self._client is None:
            self._client = GitHubClient()
        return self._client

    @property
    def cache(self) -> GitHubCache:
        """Get or create cache."""
        if self._cache is None:
            self._cache = GitHubCache()
        return self._cache

    def _resolve_repo(self, repo: str) -> str:
        """Resolve 'self' alias to actual repo."""
        if repo.lower() == "self":
            return GITHUB_SELF_REPO
        return repo

    async def cleanup(self):
        """Clean up resources."""
        if self._client:
            await self._client.close()

    # ============== Tools ==============

    @tool(
        name="read_github_file",
        description="Read a file from a GitHub repository. Use 'self' as repo to read from the agent's own codebase.",
        category=ToolCategory.DATA_ACCESS,
    )
    async def read_github_file(
        self,
        repo: str,
        path: str,
        ref: str = "main",
    ) -> ToolResult:
        """Read a file from GitHub.

        Args:
            repo: Repository in 'owner/repo' format, or 'self' for agent's codebase
            path: Path to file within the repository
            ref: Branch, tag, or commit SHA (default: main)

        Returns:
            File content with path header
        """
        repo = self._resolve_repo(repo)
        if ref == "main":
            ref = GITHUB_DEFAULT_BRANCH

        # Check cache first
        cached = await self.cache.get(repo, path, ref)
        if cached:
            return ToolResult.ok(
                f"# {path} (cached)\n\n{cached.content}",
                data={"repo": repo, "path": path, "ref": ref, "cached": True},
            )

        # Fetch from GitHub
        try:
            content = await self.client.get_file_content(repo, path, ref)
            # Cache it
            await self.cache.set(content)
            return ToolResult.ok(
                f"# {path}\n\n{content.content}",
                data={"repo": repo, "path": path, "ref": ref, "cached": False},
            )
        except GitHubClientError as e:
            return ToolResult.failed(error=f"Error reading {path}: {e}")

    @tool(
        name="list_github_files",
        description="List files in a GitHub repository directory. Use 'self' as repo for agent's codebase.",
        category=ToolCategory.DATA_ACCESS,
    )
    async def list_github_files(
        self,
        repo: str,
        path: str = "",
        ref: str = "main",
        recursive: bool = False,
    ) -> ToolResult:
        """List files in a directory.

        Args:
            repo: Repository in 'owner/repo' format, or 'self'
            path: Directory path (empty for root)
            ref: Branch, tag, or commit SHA
            recursive: If true, list all files recursively

        Returns:
            Formatted file listing
        """
        repo = self._resolve_repo(repo)
        if ref == "main":
            ref = GITHUB_DEFAULT_BRANCH

        try:
            if recursive:
                files = await self.client.get_tree(repo, ref, recursive=True)
                # Filter by path prefix if specified
                if path:
                    files = [f for f in files if f.path.startswith(path)]
            else:
                files = await self.client.list_directory(repo, path, ref)

            # Format output
            lines = [f"# Files in {repo}:{path or '/'}\n"]

            for f in sorted(files, key=lambda x: (not x.is_dir(), x.path)):
                if f.is_dir():
                    lines.append(f"\U0001f4c1 {f.path}/")
                else:
                    size = f"{f.size:,}" if f.size else "?"
                    lines.append(f"\U0001f4c4 {f.path} ({size} bytes)")

            return ToolResult.ok(
                "\n".join(lines),
                data={"repo": repo, "path": path, "ref": ref, "count": len(files)},
            )
        except GitHubClientError as e:
            return ToolResult.failed(error=f"Error listing {path}: {e}")

    @tool(
        name="search_github_code",
        description="Search for code in GitHub repositories. Use 'self' to search agent's codebase.",
        category=ToolCategory.DATA_ACCESS,
    )
    async def search_github_code(
        self,
        query: str,
        repo: Optional[str] = None,
        path: Optional[str] = None,
        extension: Optional[str] = None,
        max_results: int = 20,
    ) -> ToolResult:
        """Search for code in GitHub.

        Args:
            query: Search query
            repo: Limit to specific repo (optional, use 'self' for agent)
            path: Limit to path prefix (optional)
            extension: Limit to file extension (e.g., 'py')
            max_results: Maximum results (default 20)

        Returns:
            Formatted search results
        """
        if repo:
            repo = self._resolve_repo(repo)

        try:
            results = await self.client.search_code(
                query, repo=repo, path=path, extension=extension, max_results=max_results
            )

            if not results:
                return ToolResult.ok(
                    f"No results found for: {query}",
                    data={"query": query, "count": 0},
                )

            lines = [f"# Search results for: {query}\n"]

            for r in results:
                lines.append(f"\n## {r.repo}: {r.path}")
                lines.append(f"[View on GitHub]({r.html_url})")

                # Include text matches if available
                for match in r.text_matches[:2]:
                    fragment = match.get("fragment", "")
                    if fragment:
                        lines.append(f"\n```\n{fragment}\n```")

            return ToolResult.ok(
                "\n".join(lines),
                data={"query": query, "count": len(results)},
            )
        except GitHubClientError as e:
            return ToolResult.failed(error=f"Search error: {e}")

    @tool(
        name="get_code_definition",
        description="Get a function or class definition from a Python file. Uses AST for accurate extraction.",
        category=ToolCategory.DATA_ACCESS,
    )
    async def get_code_definition(
        self,
        repo: str,
        path: str,
        name: str,
        ref: str = "main",
    ) -> ToolResult:
        """Get a specific function or class definition.

        Args:
            repo: Repository or 'self'
            path: Path to Python file
            name: Function or class name
            ref: Branch, tag, or commit SHA

        Returns:
            Definition with signature, docstring, and source
        """
        repo = self._resolve_repo(repo)
        if ref == "main":
            ref = GITHUB_DEFAULT_BRANCH

        if not path.endswith(".py"):
            return ToolResult.failed(error="AST analysis only supports Python files (.py)")

        # Get file content
        try:
            cached = await self.cache.get(repo, path, ref)
            if cached:
                content = cached.content
            else:
                file_content = await self.client.get_file_content(repo, path, ref)
                await self.cache.set(file_content)
                content = file_content.content
        except GitHubClientError as e:
            return ToolResult.failed(error=f"Error reading {path}: {e}")

        # Parse and find definition
        analyzer = ASTAnalyzer(content, path)
        defn = analyzer.get_definition(name)

        if not defn:
            # List available definitions
            all_defs = analyzer.get_definitions()
            available = [d.name for d in all_defs[:20]]
            return ToolResult.ok(
                f"Definition '{name}' not found in {path}.\n\nAvailable: {', '.join(available)}",
                data={"repo": repo, "path": path, "name": name, "found": False, "available": available},
            )

        return ToolResult.ok(
            f"""# {defn.type.title()}: {defn.name}

**File:** {path}
**Lines:** {defn.start_line}-{defn.end_line}
**Signature:** `{defn.signature}`

## Docstring
{defn.docstring or "(no docstring)"}

## Source
```python
{defn.source}
```""",
            data={
                "repo": repo,
                "path": path,
                "name": defn.name,
                "type": defn.type,
                "start_line": defn.start_line,
                "end_line": defn.end_line,
                "found": True,
            },
        )

    @tool(
        name="list_code_definitions",
        description="List all functions and classes in a Python file.",
        category=ToolCategory.DATA_ACCESS,
    )
    async def list_code_definitions(
        self,
        repo: str,
        path: str,
        ref: str = "main",
    ) -> ToolResult:
        """List all definitions in a Python file.

        Args:
            repo: Repository or 'self'
            path: Path to Python file
            ref: Branch, tag, or commit SHA

        Returns:
            Organized list of all definitions
        """
        repo = self._resolve_repo(repo)
        if ref == "main":
            ref = GITHUB_DEFAULT_BRANCH

        if not path.endswith(".py"):
            return ToolResult.failed(error="AST analysis only supports Python files (.py)")

        # Get file content
        try:
            cached = await self.cache.get(repo, path, ref)
            if cached:
                content = cached.content
            else:
                file_content = await self.client.get_file_content(repo, path, ref)
                await self.cache.set(file_content)
                content = file_content.content
        except GitHubClientError as e:
            return ToolResult.failed(error=f"Error reading {path}: {e}")

        # Parse
        analyzer = ASTAnalyzer(content, path)
        definitions = analyzer.get_definitions()

        if not definitions:
            return ToolResult.ok(
                f"No function or class definitions found in {path}",
                data={"repo": repo, "path": path, "count": 0},
            )

        lines = [f"# Definitions in {path}\n"]

        # Group by type
        classes = [d for d in definitions if d.type == "class"]
        functions = [d for d in definitions if d.type == "function"]
        methods = [d for d in definitions if d.type == "method"]

        if classes:
            lines.append("\n## Classes")
            for d in classes:
                lines.append(f"- `{d.signature}` (lines {d.start_line}-{d.end_line})")

        if functions:
            lines.append("\n## Functions")
            for d in functions:
                lines.append(f"- `{d.signature}` (lines {d.start_line}-{d.end_line})")

        if methods:
            lines.append(f"\n## Methods ({len(methods)} total)")
            for d in methods[:30]:  # Limit output
                lines.append(f"- `{d.name}` (line {d.start_line})")
            if len(methods) > 30:
                lines.append(f"  ... and {len(methods) - 30} more")

        return ToolResult.ok(
            "\n".join(lines),
            data={
                "repo": repo,
                "path": path,
                "count": len(definitions),
                "classes": len(classes),
                "functions": len(functions),
                "methods": len(methods),
            },
        )

    @tool(
        name="get_self_repo_info",
        description="Get information about the agent's own source repository.",
        category=ToolCategory.DATA_ACCESS,
    )
    async def get_self_repo_info(self) -> ToolResult:
        """Get info about the agent's own repository.

        Returns:
            Repository metadata
        """
        repo = GITHUB_SELF_REPO

        try:
            info = await self.client.get_repo_info(repo)

            return ToolResult.ok(
                f"""# Agent Source Repository

**Repository:** {info.get('full_name')}
**Description:** {info.get('description', 'N/A')}
**Default Branch:** {info.get('default_branch', 'main')}
**Visibility:** {info.get('visibility', 'unknown')}
**Language:** {info.get('language', 'Python')}
**Size:** {info.get('size', 0):,} KB
**URL:** {info.get('html_url')}

## Stats
- Stars: {info.get('stargazers_count', 0)}
- Forks: {info.get('forks_count', 0)}
- Open Issues: {info.get('open_issues_count', 0)}
- Last Updated: {info.get('updated_at', 'unknown')}

Use `list_source_components` to see the feature components that make up this agent.""",
                data={
                    "repo": info.get("full_name", repo),
                    "default_branch": info.get("default_branch"),
                    "visibility": info.get("visibility"),
                    "open_issues_count": info.get("open_issues_count"),
                    "url": info.get("html_url"),
                },
            )
        except GitHubClientError as e:
            return ToolResult.failed(error=f"Error getting repo info: {e}")

    @tool(
        name="get_github_repo_info",
        description=(
            "Get metadata for any accessible GitHub repository, including "
            "visibility, default branch, description, and open issue count. "
            "Use 'self' as repo for the agent's own repo."
        ),
        category=ToolCategory.DATA_ACCESS,
    )
    async def get_github_repo_info(self, repo: str = "self") -> ToolResult:
        """Get metadata for an arbitrary repository.

        Args:
            repo: Repository in 'owner/repo' format, or 'self' for the
                agent's own repo.

        Returns:
            Structured repository metadata including visibility.
        """
        repo = self._resolve_repo(repo)

        try:
            info = await self.client.get_repo_info(repo)
        except GitHubClientError as e:
            return ToolResult.failed(error=f"Error getting repo info for {repo}: {e}")

        full_name = info.get("full_name", repo)
        visibility = info.get("visibility", "unknown")
        default_branch = info.get("default_branch")
        description = info.get("description")
        url = info.get("html_url")
        open_issues_count = info.get("open_issues_count")
        updated_at = info.get("updated_at")

        lines = [
            f"# {full_name}\n",
            f"**Visibility:** {visibility}",
            f"**Default Branch:** {default_branch or 'unknown'}",
            f"**Description:** {description or 'N/A'}",
            f"**URL:** {url}",
            f"**Open Issues:** {open_issues_count if open_issues_count is not None else 'unknown'}",
            f"**Last Updated:** {updated_at or 'unknown'}",
        ]

        return ToolResult.ok(
            "\n".join(lines),
            data={
                "repo": full_name,
                "visibility": visibility,
                "default_branch": default_branch,
                "description": description,
                "url": url,
                "open_issues_count": open_issues_count,
                "updated_at": updated_at,
            },
        )

    @tool(
        name="list_source_components",
        description="List all feature components in the agent's source code with their manifests.",
        category=ToolCategory.DATA_ACCESS,
    )
    async def list_source_components(self, include_files: bool = False) -> ToolResult:
        """List all feature components.

        Args:
            include_files: Include file listings for each component

        Returns:
            Formatted component list with manifests
        """
        repo = GITHUB_SELF_REPO
        ref = GITHUB_DEFAULT_BRANCH

        # Get features directory listing
        try:
            files = await self.client.list_directory(repo, GITHUB_SELF_FEATURES_ROOT, ref)
        except GitHubClientError as e:
            return ToolResult.failed(error=f"Could not access features directory: {e}")

        components = []

        for f in files:
            if f.is_dir() and not f.name.startswith("_"):
                # Try to get component.yaml
                manifest = None
                try:
                    manifest_content = await self.client.get_file_content(
                        repo, f"{GITHUB_SELF_FEATURES_ROOT}/{f.name}/component.yaml", ref
                    )
                    manifest_data = yaml.safe_load(manifest_content.content)
                    manifest = ComponentManifest.from_dict(manifest_data, f.name)
                except GitHubClientError:
                    # No manifest, create basic info
                    manifest = ComponentManifest(
                        feature_name=f.name,
                        description="(no component.yaml)",
                    )

                component_info = {
                    "name": f.name,
                    "manifest": manifest,
                }

                if include_files:
                    # List files in component directory
                    try:
                        comp_files = await self.client.get_tree(repo, ref)
                        comp_files = [
                            cf for cf in comp_files
                            if cf.path.startswith(f"{GITHUB_SELF_FEATURES_ROOT}/{f.name}/") and cf.is_file()
                        ]
                        component_info["files"] = [cf.path for cf in comp_files]
                    except GitHubClientError:
                        component_info["files"] = []

                components.append(component_info)

        # Format output
        lines = ["# Agent Source Components\n"]

        for comp in components:
            m = comp["manifest"]
            lines.append(f"\n## {m.feature_name}")
            lines.append(f"**Description:** {m.description}")
            lines.append(f"**Version:** {m.version}")
            lines.append(f"**Entry Point:** {GITHUB_SELF_FEATURES_ROOT}/{m.feature_name}/{m.entry_point}")

            if m.tools:
                lines.append(f"**Tools:** {', '.join(m.tools)}")

            if include_files and comp.get("files"):
                lines.append("\n**Files:**")
                for path in comp["files"][:20]:
                    lines.append(f"  - {path}")
                if len(comp["files"]) > 20:
                    lines.append(f"  ... and {len(comp['files']) - 20} more")

        return ToolResult.ok(
            "\n".join(lines),
            data={"repo": repo, "component_count": len(components)},
        )

    @tool(
        name="get_component_source",
        description="Get all source files for a specific feature component.",
        category=ToolCategory.DATA_ACCESS,
    )
    async def get_component_source(
        self,
        component: str,
        include_content: bool = False,
    ) -> ToolResult:
        """Get source files for a component.

        Args:
            component: Component name (e.g., 'compute', 'security', 'github')
            include_content: Include file contents (warning: may be large)

        Returns:
            Component files and optionally contents
        """
        repo = GITHUB_SELF_REPO
        ref = GITHUB_DEFAULT_BRANCH

        component_path = f"{GITHUB_SELF_FEATURES_ROOT}/{component}"

        # Get all files in component
        try:
            all_files = await self.client.get_tree(repo, ref)
            comp_files = [
                f for f in all_files
                if f.path.startswith(component_path + "/") and f.is_file()
            ]
        except GitHubClientError as e:
            return ToolResult.failed(error=f"Could not access component '{component}': {e}")

        if not comp_files:
            return ToolResult.ok(
                f"Component '{component}' not found or has no files",
                data={"component": component, "file_count": 0},
            )

        lines = [f"# Component: {component}\n"]
        lines.append(f"**Path:** {component_path}")
        lines.append(f"**Files:** {len(comp_files)}")

        # Try to get manifest
        try:
            manifest_content = await self.client.get_file_content(
                repo, f"{component_path}/component.yaml", ref
            )
            lines.append("\n## Manifest (component.yaml)")
            lines.append(f"```yaml\n{manifest_content.content}\n```")
        except GitHubClientError:
            lines.append("\n*No component.yaml manifest*")

        lines.append("\n## Files")

        for f in sorted(comp_files, key=lambda x: x.path):
            rel_path = f.path[len(component_path) + 1:]
            lines.append(f"\n### {rel_path}")

            if include_content and f.path.endswith(".py"):
                try:
                    content = await self.client.get_file_content(repo, f.path, ref)
                    await self.cache.set(content)
                    lines.append(f"```python\n{content.content}\n```")
                except GitHubClientError as e:
                    lines.append(f"*Could not read: {e}*")
            else:
                lines.append(f"*Size: {f.size:,} bytes*")

        return ToolResult.ok(
            "\n".join(lines),
            data={"component": component, "path": component_path, "file_count": len(comp_files)},
        )

    @tool(
        name="invalidate_github_cache",
        description="Invalidate cached GitHub content to force fresh fetch.",
        category=ToolCategory.DATA_ACCESS,
    )
    async def invalidate_github_cache(
        self,
        repo: str,
        path: Optional[str] = None,
    ) -> ToolResult:
        """Invalidate cache entries.

        Args:
            repo: Repository to invalidate (or 'self')
            path: Specific path to invalidate (optional)

        Returns:
            Confirmation message
        """
        repo = self._resolve_repo(repo)

        await self.cache.invalidate(repo, path=path)

        if path:
            return ToolResult.ok(
                f"Invalidated cache for {repo}:{path}",
                data={"repo": repo, "path": path},
            )
        return ToolResult.ok(
            f"Invalidated all cache for {repo}",
            data={"repo": repo, "path": None},
        )

    # --- Stalled-work tools ---

    @tool(
        name="scan_stale_work",
        description=(
            "Scan one or more repos for stalled or blocking work: a red default "
            "branch (failed CI), claimed issues gone quiet, and stale open PRs. "
            "The detection half of the proactive rescue loop (sovereign #1523)."
        ),
        category=ToolCategory.DATA_ACCESS,
    )
    async def scan_stale_work(
        self,
        repos: str = "fleet",
        stale_days: int = 3,
    ) -> ToolResult:
        """Detect stalled/blocking work across repos.

        Args:
            repos: Comma-separated 'owner/repo' slugs, 'self' for the agent's
                own repo, or 'fleet' (default) for the configured fleet
                (``GITHUB_FLEET_REPOS``, falling back to 'self').
            stale_days: An issue/PR counts as stale after this many days with
                no update (default 3).

        Returns:
            Structured findings sorted by severity (high first).
        """
        slugs = self._resolve_repos(repos)
        if not slugs:
            return ToolResult.failed(error="No repositories given")

        findings, errors = await self._scan_repos(slugs, stale_days)

        high = sum(1 for f in findings if f["severity"] == "high")
        summary = (
            f"{len(findings)} stalled-work finding(s) across {len(slugs)} repo(s)"
            f" ({high} high)."
        )
        if errors:
            summary += f" {len(errors)} repo(s) had scan errors."
        return ToolResult.ok(
            summary,
            data={
                "findings": findings,
                "errors": errors,
                "repos_scanned": slugs,
                "stale_days": stale_days,
            },
        )

    def _resolve_repos(self, repos: str) -> list[str]:
        """Resolve a repos spec to concrete slugs.

        'fleet' => configured fleet (or self); otherwise a comma-separated list
        of slugs / 'self' aliases.
        """
        if repos.strip().lower() == "fleet":
            return _configured_fleet_repos()
        return [self._resolve_repo(r.strip()) for r in repos.split(",") if r.strip()]

    async def _scan_repos(
        self, slugs: list[str], stale_days: int
    ) -> tuple[list[dict], list[dict]]:
        """Core sweep shared by the tool and the fleet_stalled_sweep source.

        Returns ``(findings, errors)``; findings are dicts sorted by severity.
        """
        now = datetime.now(timezone.utc)
        severity_order = {"high": 0, "medium": 1, "low": 2}
        findings: list[dict] = []
        errors: list[dict] = []

        for repo in slugs:
            try:
                info = await self.client.get_repo_info(repo)
                branch = str(info.get("default_branch") or "main")
                # Oldest-updated first so the stalest claimed issues land on the
                # first page rather than being hidden behind a large backlog.
                issues = await self.client.list_issues(
                    repo, state="open", per_page=100, sort="updated", direction="asc",
                )
                prs = await self.client.list_pull_requests(repo, state="open")
            except GitHubClientError as e:
                errors.append({"repo": repo, "error": str(e)})
                continue

            # CI status is best-effort: Actions may be disabled, or the token
            # may have Issues/PRs read but not Actions read. A failure here must
            # NOT suppress the issue/PR findings — just skip the red-branch check.
            latest_run = None
            try:
                runs = await self.client.list_workflow_runs(repo, branch=branch, per_page=1)
                latest_run = runs[0] if runs else None
            except GitHubClientError as e:
                errors.append(
                    {"repo": repo, "error": f"CI status unavailable: {e}", "partial": True}
                )

            items = classify_stale_work(
                repo,
                issues=issues,
                pull_requests=prs,
                latest_default_run=latest_run,
                default_branch=branch,
                now=now,
                stale_days=stale_days,
            )
            findings.extend(asdict(it) for it in items)

        findings.sort(key=lambda f: (severity_order.get(f["severity"], 9), f["repo"], f["ref"]))
        return findings, errors

    # --- Issue tools ---

    @tool(
        name="list_github_issues",
        description="List issues in a GitHub repository. Filters out pull requests.",
        category=ToolCategory.DATA_ACCESS,
    )
    async def list_github_issues(
        self,
        repo: str = "self",
        state: str = "open",
        labels: Optional[str] = None,
        max_results: int = 30,
    ) -> ToolResult:
        """List issues in a repository.

        Args:
            repo: Repository in 'owner/repo' format, or 'self' for agent's own repo
            state: Issue state filter ('open', 'closed', 'all')
            labels: Comma-separated label names to filter by
            max_results: Maximum number of issues to return (max 100)

        Returns:
            Formatted issue list
        """
        repo = self._resolve_repo(repo)

        try:
            label_list = [l.strip() for l in labels.split(",")] if labels else None
            issues = await self.client.list_issues(
                repo, state=state, labels=label_list, per_page=max_results,
            )
        except GitHubClientError as e:
            return ToolResult.failed(error=f"Could not list issues: {e}")

        if not issues:
            return ToolResult.ok(
                f"No {state} issues found in {repo}",
                data={"repo": repo, "state": state, "count": 0},
            )

        lines = [f"# Issues in {repo} ({state})\n"]
        for issue in issues:
            number = issue.get("number")
            title = issue.get("title", "")
            issue_labels = [l["name"] for l in issue.get("labels", [])]
            assignees = [a["login"] for a in issue.get("assignees", [])]
            updated = issue.get("updated_at", "")[:10]

            line = f"- **#{number}** {title}"
            if issue_labels:
                line += f"  [{', '.join(issue_labels)}]"
            if assignees:
                line += f"  @{', @'.join(assignees)}"
            line += f"  (updated {updated})"
            lines.append(line)

        lines.append(f"\n*{len(issues)} issue(s) shown*")
        return ToolResult.ok(
            "\n".join(lines),
            data={"repo": repo, "state": state, "count": len(issues)},
        )

    @tool(
        name="get_github_issue",
        description="Get details of a specific GitHub issue by number.",
        category=ToolCategory.DATA_ACCESS,
    )
    async def get_github_issue(
        self,
        issue_number: int,
        repo: str = "self",
    ) -> ToolResult:
        """Get a specific issue.

        Args:
            issue_number: Issue number
            repo: Repository in 'owner/repo' format, or 'self' for agent's own repo

        Returns:
            Formatted issue details
        """
        repo = self._resolve_repo(repo)

        try:
            issue = await self.client.get_issue(repo, issue_number)
        except GitHubClientError as e:
            return ToolResult.failed(error=f"Could not get issue #{issue_number}: {e}")

        title = issue.get("title", "")
        state = issue.get("state", "")
        body = issue.get("body", "") or "(no description)"
        author = issue.get("user", {}).get("login", "unknown")
        created = issue.get("created_at", "")[:10]
        updated = issue.get("updated_at", "")[:10]
        issue_labels = [l["name"] for l in issue.get("labels", [])]
        assignees = [a["login"] for a in issue.get("assignees", [])]
        milestone = issue.get("milestone", {})
        milestone_name = milestone.get("title") if milestone else None
        comments_count = issue.get("comments", 0)

        lines = [
            f"# #{issue_number}: {title}\n",
            f"**State:** {state}",
            f"**Author:** @{author}",
            f"**Created:** {created} | **Updated:** {updated}",
        ]
        if issue_labels:
            lines.append(f"**Labels:** {', '.join(issue_labels)}")
        if assignees:
            lines.append(f"**Assignees:** {', '.join('@' + a for a in assignees)}")
        if milestone_name:
            lines.append(f"**Milestone:** {milestone_name}")
        lines.append(f"**Comments:** {comments_count}")
        lines.append(f"\n---\n\n{body}")

        return ToolResult.ok(
            "\n".join(lines),
            data={
                "repo": repo,
                "number": issue_number,
                "title": title,
                "state": state,
                "author": author,
                "labels": issue_labels,
                "assignees": assignees,
                "milestone": milestone_name,
                "comments_count": comments_count,
                "url": issue.get("html_url"),
            },
        )

    @tool(
        name="get_github_issue_comments",
        description="Get comments on a specific GitHub issue.",
        category=ToolCategory.DATA_ACCESS,
    )
    async def get_github_issue_comments(
        self,
        issue_number: int,
        repo: str = "self",
        max_results: int = 30,
    ) -> ToolResult:
        """Get comments on an issue.

        Args:
            issue_number: Issue number
            repo: Repository in 'owner/repo' format, or 'self' for agent's own repo
            max_results: Maximum number of comments to return

        Returns:
            Formatted comment list
        """
        repo = self._resolve_repo(repo)

        try:
            comments = await self.client.get_issue_comments(
                repo, issue_number, per_page=max_results,
            )
        except GitHubClientError as e:
            return ToolResult.failed(error=f"Could not get comments for issue #{issue_number}: {e}")

        if not comments:
            return ToolResult.ok(
                f"No comments on issue #{issue_number} in {repo}",
                data={"repo": repo, "number": issue_number, "count": 0},
            )

        lines = [f"# Comments on #{issue_number} in {repo}\n"]
        for comment in comments:
            author = comment.get("user", {}).get("login", "unknown")
            created = comment.get("created_at", "")[:10]
            body = comment.get("body", "")

            lines.append(f"## @{author} ({created})\n")
            lines.append(body)
            lines.append("")

        lines.append(f"\n*{len(comments)} comment(s)*")
        return ToolResult.ok(
            "\n".join(lines),
            data={"repo": repo, "number": issue_number, "count": len(comments)},
        )

    # --- Write tools (issues, PRs, comments, labels) ---
    #
    # These let the agent close its own loops: file a follow-up ticket
    # surfaced during diagnosis, open a rescue PR for an orphaned branch,
    # comment on its own filed issues, edit lifecycle labels, close
    # completed issues, and merge an own-PR that has reached green CI.
    #
    # Each is a thin wrapper around the client write methods with the
    # ToolResult envelope and consistent failure shaping. The PRE_TOOL_USE
    # hook chain gates these per-agent at the kestrel-sovereign layer —
    # no enforcement here, by design.

    @tool(
        name="create_github_issue",
        description=(
            "File a new GitHub issue. Returns the URL and number on success. "
            "Use 'self' as repo for the agent's own repo."
        ),
        category=ToolCategory.SYSTEM,
    )
    async def create_github_issue(
        self,
        title: str,
        body: str,
        repo: str = "self",
        labels: Optional[str] = None,
        assignees: Optional[str] = None,
    ) -> ToolResult:
        """Open a new issue.

        Args:
            title: Issue title.
            body: Issue body (markdown).
            repo: ``owner/repo`` or ``self``.
            labels: Comma-separated labels.
            assignees: Comma-separated GitHub usernames.
        """
        repo = self._resolve_repo(repo)
        # Filter empty CSV entries — a trailing comma or extra space would
        # otherwise send an empty string to GitHub and trigger a 422 from
        # validation (codex round 1 P2).
        parsed_labels = (
            [s.strip() for s in labels.split(",") if s.strip()] if labels else None
        )
        parsed_assignees = (
            [s.strip() for s in assignees.split(",") if s.strip()] if assignees else None
        )
        try:
            issue = await self.client.create_issue(
                repo,
                title=title,
                body=body,
                labels=parsed_labels or None,
                assignees=parsed_assignees or None,
            )
        except GitHubClientError as e:
            return ToolResult.failed(error=f"Could not create issue: {e}")

        return ToolResult.ok(
            f"Filed {repo}#{issue.get('number')}: {issue.get('html_url')}",
            data={
                "repo": repo,
                "number": issue.get("number"),
                "url": issue.get("html_url"),
            },
        )

    @tool(
        name="add_github_issue_comment",
        description=(
            "Post a comment on a GitHub issue or pull request. "
            "Use 'self' as repo for the agent's own repo."
        ),
        category=ToolCategory.SYSTEM,
    )
    async def add_github_issue_comment(
        self,
        issue_number: int,
        body: str,
        repo: str = "self",
    ) -> ToolResult:
        """Add a comment to an issue or PR."""
        repo = self._resolve_repo(repo)
        try:
            comment = await self.client.add_issue_comment(repo, issue_number, body)
        except GitHubClientError as e:
            return ToolResult.failed(
                error=f"Could not comment on #{issue_number}: {e}",
            )
        return ToolResult.ok(
            f"Commented on {repo}#{issue_number}",
            data={
                "repo": repo,
                "number": issue_number,
                "comment_id": comment.get("id"),
                "url": comment.get("html_url"),
            },
        )

    @tool(
        name="add_github_label",
        description=(
            "Add one or more labels to an issue or PR. Comma-separated. "
            "Use 'self' as repo for the agent's own repo."
        ),
        category=ToolCategory.SYSTEM,
    )
    async def add_github_label(
        self,
        issue_number: int,
        labels: str,
        repo: str = "self",
    ) -> ToolResult:
        """Add labels to an issue or PR (existing labels preserved)."""
        repo = self._resolve_repo(repo)
        label_list = [s.strip() for s in labels.split(",") if s.strip()]
        if not label_list:
            return ToolResult.failed(error="No labels provided")
        try:
            updated = await self.client.add_labels(repo, issue_number, label_list)
        except GitHubClientError as e:
            return ToolResult.failed(
                error=f"Could not add labels to #{issue_number}: {e}",
            )
        return ToolResult.ok(
            f"Added {len(label_list)} label(s) to {repo}#{issue_number}",
            data={
                "repo": repo,
                "number": issue_number,
                "added": label_list,
                "current": [l.get("name") for l in updated],
            },
        )

    @tool(
        name="remove_github_label",
        description=(
            "Remove a single label from an issue or PR. Idempotent — "
            "succeeds whether or not the label was present. "
            "Use 'self' as repo for the agent's own repo."
        ),
        category=ToolCategory.SYSTEM,
    )
    async def remove_github_label(
        self,
        issue_number: int,
        label: str,
        repo: str = "self",
    ) -> ToolResult:
        """Remove a single label from an issue or PR."""
        repo = self._resolve_repo(repo)
        try:
            await self.client.remove_label(repo, issue_number, label)
        except GitHubClientError as e:
            return ToolResult.failed(
                error=f"Could not remove label {label!r} from #{issue_number}: {e}",
            )
        return ToolResult.ok(
            f"Removed label {label!r} from {repo}#{issue_number}",
            data={"repo": repo, "number": issue_number, "removed": label},
        )

    @tool(
        name="close_github_issue",
        description=(
            "Close a GitHub issue. ``state_reason`` is one of "
            "``completed`` (default), ``not_planned``. "
            "Use 'self' as repo for the agent's own repo."
        ),
        category=ToolCategory.SYSTEM,
    )
    async def close_github_issue(
        self,
        issue_number: int,
        repo: str = "self",
        state_reason: str = "completed",
    ) -> ToolResult:
        """Close an issue with an explicit reason."""
        repo = self._resolve_repo(repo)
        try:
            issue = await self.client.update_issue(
                repo,
                issue_number,
                state="closed",
                state_reason=state_reason,
            )
        except GitHubClientError as e:
            return ToolResult.failed(
                error=f"Could not close #{issue_number}: {e}",
            )
        return ToolResult.ok(
            f"Closed {repo}#{issue_number} ({state_reason})",
            data={
                "repo": repo,
                "number": issue_number,
                "state": issue.get("state"),
                "state_reason": issue.get("state_reason"),
            },
        )

    @tool(
        name="reopen_github_issue",
        description=(
            "Reopen a closed GitHub issue. "
            "Use 'self' as repo for the agent's own repo."
        ),
        category=ToolCategory.SYSTEM,
    )
    async def reopen_github_issue(
        self,
        issue_number: int,
        repo: str = "self",
    ) -> ToolResult:
        """Reopen a closed issue."""
        repo = self._resolve_repo(repo)
        try:
            issue = await self.client.update_issue(
                repo,
                issue_number,
                state="open",
                state_reason="reopened",
            )
        except GitHubClientError as e:
            return ToolResult.failed(
                error=f"Could not reopen #{issue_number}: {e}",
            )
        return ToolResult.ok(
            f"Reopened {repo}#{issue_number}",
            data={
                "repo": repo,
                "number": issue_number,
                "state": issue.get("state"),
            },
        )

    @tool(
        name="create_github_pull_request",
        description=(
            "Open a pull request on a GitHub repository. "
            "Use 'self' as repo for the agent's own repo."
        ),
        category=ToolCategory.SYSTEM,
    )
    async def create_github_pull_request(
        self,
        title: str,
        head: str,
        base: str,
        body: str = "",
        repo: str = "self",
        draft: bool = False,
    ) -> ToolResult:
        """Open a pull request.

        Args:
            title: PR title.
            head: Branch carrying changes (same-repo: branch name;
                cross-fork: ``owner:branch``).
            base: Target branch on the repo.
            body: PR body (markdown).
            repo: ``owner/repo`` or ``self``.
            draft: Open as a draft PR.
        """
        repo = self._resolve_repo(repo)
        try:
            pr = await self.client.create_pull_request(
                repo,
                title=title,
                head=head,
                base=base,
                body=body,
                draft=draft,
            )
        except GitHubClientError as e:
            return ToolResult.failed(error=f"Could not open PR: {e}")
        return ToolResult.ok(
            f"Opened {repo}#{pr.get('number')}: {pr.get('html_url')}",
            data={
                "repo": repo,
                "number": pr.get("number"),
                "url": pr.get("html_url"),
                "head_sha": (pr.get("head") or {}).get("sha"),
                "draft": pr.get("draft"),
            },
        )

    @tool(
        name="merge_github_pull_request",
        description=(
            "Merge a pull request once CI/review allow. "
            "``merge_method`` is ``squash`` (default), ``merge``, or ``rebase``. "
            "Pass ``sha`` to refuse the merge if the PR head has moved. "
            "Use 'self' as repo for the agent's own repo."
        ),
        category=ToolCategory.SYSTEM,
    )
    async def merge_github_pull_request(
        self,
        pull_number: int,
        repo: str = "self",
        merge_method: str = "squash",
        commit_title: Optional[str] = None,
        commit_message: Optional[str] = None,
        sha: Optional[str] = None,
    ) -> ToolResult:
        """Merge a PR with the chosen method."""
        repo = self._resolve_repo(repo)
        try:
            result = await self.client.merge_pull_request(
                repo,
                pull_number,
                merge_method=merge_method,
                commit_title=commit_title,
                commit_message=commit_message,
                sha=sha,
            )
        except GitHubClientError as e:
            return ToolResult.failed(
                error=f"Could not merge PR #{pull_number}: {e}",
            )
        return ToolResult.ok(
            f"Merged {repo}#{pull_number} ({merge_method}) at {result.get('sha','?')[:10]}",
            data={
                "repo": repo,
                "number": pull_number,
                "merge_method": merge_method,
                "merge_commit_sha": result.get("sha"),
                "merged": result.get("merged"),
            },
        )
