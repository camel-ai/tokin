import tokin


def test_version_is_populated():
    # hatch-vcs derives this from git at build time; an empty string means the
    # build saw no tags/history and the wheel would ship an unusable version.
    assert tokin.__version__
