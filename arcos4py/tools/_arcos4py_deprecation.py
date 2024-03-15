import warnings


def handle_deprecated_params(param_mapping, **kwargs):
    """A utility function to handle deprecated parameters in **kwargs.

    Arguments:
    param_mapping: A dict mapping old parameter names to new ones.
    **kwargs: The **kwargs from the calling function.

    Returns:
    Updated **kwargs with deprecated parameters handled.
    """
    updated_kwargs = kwargs.copy()  # Make a copy to avoid modifying the original **kwargs
    for old_param, new_param in param_mapping.items():
        if old_param in updated_kwargs:
            # Issue a deprecation warning
            warnings.warn(
                f"The '{old_param}' parameter is deprecated and will be removed in a future version.\
                Please use '{new_param}' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            if new_param in updated_kwargs:
                # If both new and old parameters are provided, overwrite the new with the old
                warnings.warn(
                    f"Both '{old_param}' and '{new_param}' are provided. '{old_param}' will take precedence over\
                         '{new_param}'. Consider using only '{new_param}' in future.",
                    DeprecationWarning,
                    stacklevel=3,
                )
            updated_kwargs[new_param] = updated_kwargs.pop(old_param)
    return updated_kwargs
