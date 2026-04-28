import re

with open('README.md', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Replace operatorname with text in block math
text = text.replace(r'\operatorname{', r'\text{')

# 2. Replace the inline problem variables with code blocks instead of math mode.
text = text.replace(r'$\text{relation\_strength}$', r'`relation_strength`')
text = text.replace(r'$\text{intensity\_score}$', r'`intensity_score`')

text = text.replace('分别对 elation_strength 和 \\intensity_score 取均值、最大值：', '分别对 `relation_strength` 和 `intensity_score` 取均值、最大值：')


with open('README.md', 'w', encoding='utf-8') as f:
    f.write(text)

print("Done fixing README.md")
