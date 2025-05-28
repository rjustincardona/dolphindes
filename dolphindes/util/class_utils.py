def check_attributes(self, *attrs):
    ### maybe this helper function for lazy initialization can also be useful elsewhere?
    missing = [attr for attr in attrs if getattr(self, attr) is None]
    if missing:
        raise AttributeError(f"{', '.join(missing)} undefined.")
