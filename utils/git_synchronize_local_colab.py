import os
import dotenv
import subprocess
from pathlib import Path
from typing import List
from utils.torch_ckpt import find_root

dotenv.load_dotenv()
PROJ_NAME = os.getenv("PROJ_NAME")

PROJ_DIR = find_root(PROJ_NAME)

def check_git_available() -> bool:
    '''
    Check if git is available in the current environment.
    Called automatically on module import.

    Checks:
        1. Whether .git folder exists in PROJ_DIR
        2. Whether git command is available in PATH

    Returns:
        True if git is available, False otherwise

    Side effect:
        Prints a warning if git is not available
    '''
    # 1) Check if .git folder exists
    git_dir = Path(PROJ_DIR) / ".git"
    if not git_dir.exists():
        print(f"[WARNING] No .git folder found in {PROJ_DIR}. Git functions may not work.")
        return False

    # 2) Check if git command is available
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("[WARNING] git command is not working properly. Git functions may not work.")
            return False
    except FileNotFoundError:
        print("[WARNING] git command not found. Please install git.")
        return False

    return True


# Check git availability on module import
check_git_available()


def git_check_current_branch():
    result = subprocess.run(
        ['git', 'branch', '--show-current'],
        cwd=PROJ_DIR, capture_output=True, text=True, check=True
    ).stdout.strip()
    print(f"================= Current branch: {result} =================")
    return result

def git_check_remote_sync(branch_name:str):
    '''
    If the latest remote commit is ahead of the local commit, your current change would be crashed. 
        - So before doing work right now, you should pull the latest remote commit to prevent the crash.
    '''
    subprocess.run(
        ['git', 'fetch', 'origin'],
        cwd=PROJ_DIR, capture_output=True, text=True, check=True
    )

    local_head = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        cwd=PROJ_DIR, capture_output=True, text=True, check=True
    )

    remote_head = subprocess.run(
        ['git', 'rev-parse', f'origin/{branch_name}'],
        cwd=PROJ_DIR, capture_output=True, text=True, check=True
    )

    if local_head.stdout.strip() != remote_head.stdout.strip():
        raise RuntimeError(f"""Local and remote are not in the same commit. 
        Local commit: {local_head.stdout.strip()}
        Remote commit: {remote_head.stdout.strip()}
        Recommendation
        (1) If you didn't start to change the code, you should sync your code before change the code.\n
        (2) If you started to change the code after perceiving the remote code, after finshing your code change, you should push your code to remote.\n
        """)
    else:
        print("Local and remote are in the same commit - it's synced!")

def git_check_any_changes(branch_name: str) -> bool:
    '''
    Check "Is there any work or change from the latest commit?"

    Returns:
        True if working tree is clean (same as latest commit)
        False if there are changes (tracked or untracked)

    Side effect:
        If there are changes, prints the first changed file and "..." if more than one

    In terms of local environment work, there are n cases:

    1. When the code change started.
        1-1. Does the current commit is synced with remote commit?
             If that's not, then we need to sync because if we change without sync,
             after change there would be an error in pushing.
        1-2. The sync is right, but current working tree could be different with
             the latest commit - that means currently I'm working on it,
             and after finish that work, I need to do push.

    2. When the code change finished.
        2-1. The code change is not reflected on github(remote),
             I need to push the code.
    '''
    repo_dir = Path(PROJ_DIR)

    # 1) Check for tracked changes (staged or unstaged)
    tracked_changed = False
    result = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode == 1:
        tracked_changed = True
    elif result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"git diff --quiet failed (code={result.returncode}): {stderr}")

    # 2) Get list of untracked files
    untracked_out = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=True
    )
    untracked_files = [line for line in untracked_out.stdout.splitlines() if line.strip()]

    # 3) Get list of tracked changed files (if any)
    tracked_files: List[str] = []
    if tracked_changed:
        diff_out = subprocess.run(
            ["git", "diff", "--name-only", "HEAD", "--"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True
        )
        tracked_files = [line for line in diff_out.stdout.splitlines() if line.strip()]

    # 4) Merge and deduplicate, preserving order
    seen = set()
    merged = []
    for p in (tracked_files + untracked_files):
        if p not in seen:
            seen.add(p)
            merged.append(p)

    # 5) Report result
    if not merged:
        print("Working tree is clean - same as the latest commit!")
        return True
    else:
        print(f"Working tree has {len(merged)} changed file(s):")
        print(f"  {merged[0]}")
        if len(merged) > 1:
            print("  ...")
        return False



# Git Synchronization functions

def git_Branch_pull(branch_name:str):
    '''
    Use this function before running the code in colab.
    Use this function before changing the code in the local
    '''
    subprocess.run(
        ["git", "switch", "-C", branch_name], 
        cwd=PROJ_DIR, capture_output=True, text=True, check=True
    )
    subprocess.run(
        ['git', 'fetch', 'origin', branch_name],
        cwd=PROJ_DIR, capture_output=True, text=True, check=True
    )
    subprocess.run(
        ["git", "pull", "--ff-only", "origin/"+branch_name], 
        cwd=PROJ_DIR, capture_output=True, text=True, check=True
    )

def git_Branch_push(branch_name:str, commit_message:str, work_environemnt:str):
    '''
    Use this function when there's any changes in local or colab afther finishing your code changes
    '''
    assert work_environemnt in ["local", "colab"], "work_environemnt must be 'local' or 'colab'"
    subprocess.run(
        ["git", "switch", "-C", branch_name], 
        cwd=PROJ_DIR, capture_output=True, text=True, check=True
    )

    subprocess.run(
        ['git', 'add', '-A'],
        cwd=PROJ_DIR, capture_output=True, text=True, check=True
    )
    subprocess.run(
        ['git', 'commit', '-m', commit_message+"\nThis is pushed from {work_environemnt}"],
        cwd=PROJ_DIR, capture_output=True, text=True, check=True
    )
    subprocess.run(
        ['git', 'push', 'origin', branch_name+":"+branch_name],
        cwd=PROJ_DIR, capture_output=True, text=True, check=True
    )

