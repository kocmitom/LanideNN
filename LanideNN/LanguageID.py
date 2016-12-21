from iso639 import languages


class LanguageID(object):
    def __init__(self, name):
        self.name = name
        try:
            if len(name) == 2:
                self.lang = languages.get(part1=name)
            elif len(name) == 3:
                self.lang = languages.get(part2t=name)
            else:
                self.lang = languages.get(name=name)
        except Exception:
            self.lang = name

    def get_iso2(self):
        try:
            if len(self.lang.part1) > 0:
                return self.lang.part1
            else:
                return self.name
        except Exception:
            return self.name

    def get_iso3(self):
        try:
            return self.lang.part2t
        except Exception:
            return self.name

    def get_all(self):
        try:
            return "{0} {1} {2}".format(self.lang.name, self.lang.part1, self.lang.part2t)
        except Exception:
            return self.name