from __future__ import annotations

from numpydoc.docscrape import NumpyDocString
from plum import dispatch
from quartodoc import MdRenderer
from quartodoc import ast as qast


class Renderer(MdRenderer):
    style = "siuba"

    @dispatch
    def render(self, el):
        """General render method.
        Note: overloading of `render` enabled via plum.dispatch to allow different
        rendering behaviour for some elements.
        """
        prev_obj = getattr(self, "crnt_obj", None)
        self.crnt_obj = el
        res = super().render(el)
        self.crnt_obj = prev_obj

        return res

    @dispatch
    def render(self, el: qast.DocstringSectionSeeAlso):  # noqa: F811
        """Numpy Docstring style render method.
        Note: overloading of `render` enabled via plum.dispatch to allow different
        rendering behaviour for some elements.
        """
        lines = el.value.split("\n")

        # each entry in result has form: ([('func1', '<directive>), ...], <description>)
        parsed = NumpyDocString("")._parse_see_also(lines)

        result = []
        for funcs, description in parsed:
            links = [f"[{name}](`{self._name_to_target(name)}`)" for name, role in funcs]

            str_links = ", ".join(links)

            if description:
                str_description = "<br>".join(description)
                result.append(f"{str_links}: {str_description}")
            else:
                result.append(str_links)

        return "*\n".join(result)

    def _name_to_target(self, name: str):
        """Helper method to convert a function/class name to a full target path,
        used for Numpy Docstring style render method.
        """
        crnt_path = getattr(self.crnt_obj, "path", None)
        parent = crnt_path.rsplit(".", 1)[0] + "."
        pkg = "classifai."

        if crnt_path and not (name.startswith(pkg) or name.startswith(parent)):
            return f"{parent}{name}"
        elif not name.startswith(pkg):
            return f"{pkg}{name}"

        return name
