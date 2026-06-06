def disable_broken_torchvision():
    """
    Transformers may import optional torchvision even for text-only models.
    If torchvision is installed with a mismatched CUDA build, text model loading
    fails before TMG-Q can run. Treat torchvision as unavailable for TMG-Q scripts.
    """
    try:
        import transformers.utils as utils
        import transformers.utils.import_utils as import_utils
    except Exception:
        return

    def unavailable():
        return False

    for name in ("is_torchvision_available", "is_torchvision_v2_available"):
        old = getattr(import_utils, name, None)
        if hasattr(old, "cache_clear"):
            old.cache_clear()
        setattr(import_utils, name, unavailable)
        if hasattr(utils, name):
            setattr(utils, name, unavailable)

    if hasattr(import_utils, "BACKENDS_MAPPING"):
        for key in ("torchvision",):
            if key in import_utils.BACKENDS_MAPPING:
                _, error = import_utils.BACKENDS_MAPPING[key]
                import_utils.BACKENDS_MAPPING[key] = (unavailable, error)
