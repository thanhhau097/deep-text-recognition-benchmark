# https://stackoverflow.com/questions/19899554/unicode-range-for-japanese

# Japanese-style punctuation ( 3000 - 303f)
# Hiragana ( 3040 - 309f)
# Katakana ( 30a0 - 30ff)
# Full-width roman characters and half-width katakana ( ff00 - ffef)
# CJK unifed ideographs - Common and uncommon kanji ( 4e00 - 9faf)

# import chr
# eng_ranges = [('0020', '007E')]
# ranges = eng_ranges + [('3000', '303f'), ('3040', '309f'), ('30a0', '30ff'), ('ff00', 'ffef'), ('4e00', '9faf')]
# count = 0
# chars = ''
# for start, end in ranges:
#     for i in range(int('0x' + end, 16) - int('0x' + start, 16) + 1):
#         chars += chr(i + int('0x' + start, 16))
#         count += 1

# with open('charset.txt', 'w') as f:
#     f.write(chars)

# print('total:', count)

