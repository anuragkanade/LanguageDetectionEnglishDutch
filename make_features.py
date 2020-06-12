def is_it_dutch_accent(line):
    """
    checks if the line has characters with an accent
    list does not include circumflex or graves as they are used with loan words,
    and dutch, as well as english have loan words

    :param line: a single line of words
    :return: true if word have a character with an accent
    """
    accent_char_ord = [193, 196, 201, 203, 205, 207, 211, 214, 218, 220, 221, 225, 228, 233,
                       235, 237, 239, 243, 246, 250, 252, 253, 255]

    for ch in line:
        if ord(ch) in accent_char_ord:
            return True
    return False


def is_dutch_preposition_is_not_english_word(line):
    """
    not including words which are dutch prepositions but are also english words

    :param line: a single line of words
    :return: true if line has a dutch preposition which is not a english word, false otherwise
    """
    dutch_prepositions = ["naar", "voor", "achter", "naast", "beneden", "boven", "onder", "op", "tussen", "het midden",
                          "bij", "binnen", "buiten", "tegen", "rond", "sinds", "zonder", "na", "om"]
    for word in dutch_prepositions:
        test_term = " " + word + " "
        if line.find(test_term) != -1:
            return True

    return False


def is_english_preposition(line):
    """
    checks if the line has an english preposition

    :param line: a single line of words
    :return: true if the line has a preposition, false otherwise
    """
    english_prepositions = ["with", "from", "to", "in front of", "behind", "next to", "down", "downstairs",
                            "above", "upstairs", "below", "on top", "between", "middle", "about", "over", "near",
                            "inside", "outside", "against", "around", "since", "without", "before", "after"]
    for word in english_prepositions:
        test_term = " " + word + " "
        if line.find(test_term) != -1:
            return True
    return False


class Features:
    __slots__ = "features"

    def __init__(self):
        self.features = {"nl_article": False,
                         "nl_prepos": False,
                         "en_article": False,
                         "en_prepos": False,
                         "accent": False,
                         "als_present": False,
                         "as_present": False,
                         "dat_present": False,
                         "that_present": False,
                         "also_present": False,
                         }

    def make_features(self, line, is_test=False):
        """
        sets boolean variables in self.features

        :param line: a single line of words
        :param is_test: is required to know if the data has the language of the line at the start, test data has them
        :return: None
        """
        line = str(line)
        line = line.lower()
        if not is_test:
            lang = line[0:2]
            line = line[3:]
            if lang == "nl":
                self.features["res"] = True
            else:
                self.features["res"] = False
        self.features["sent"] = line
        if line.find(" het ") != -1 or line.find(" de ") != -1:
            self.features["nl_article"] = True
        if is_dutch_preposition_is_not_english_word(line):
            self.features["nl_prepos"] = True
        if line.find(" the ") != -1 or line.find(" a ") != -1 or line.find(" an ") != -1:
            self.features["en_article"] = True
        if is_english_preposition(line):
            self.features["en_prepos"] = True
        if is_it_dutch_accent(line):
            self.features["accent"] = True
        if line.find(" als ") != -1:
            self.features["als_present"] = True
        if line.find(" as ") != -1:
            self.features["as_present"] = True
        if line.find(" dat ") != -1:
            self.features["dat_present"] = True
        if line.find(" that ") != -1:
            self.features["that_present"] = True
        if line.find(" also ") != -1:
            self.features["also_present"] = True

    def __str__(self):
        """
        returns a __str__ value of the self.features dictionary
        :return:
        """
        return self.features.__str__()
