import unicodedata

# with help from:
# https://stackoverflow.com/questions/10294032/python-replace-typographical-quotes-dashes-etc-with-their-ascii-counterparts

quote_characters = []
for i in range(0, 0x10ffff):
    char = chr(i)

    try:
        if 'QUOTATION MARK' in unicodedata.name(char):
            quote_characters.append(char)
    except:
        pass


def normalize_quote_characters(str):
    for quote_char in quote_characters:
        str = str.replace(quote_char, '"')

    return str
