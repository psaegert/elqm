from bs4 import BeautifulSoup

from elqm.factories.preprocessing import PreprocessorFactory

'''
Textpart of the Document 32023L1791...:
Consequently, the energy efficiency first principle should help increase the efficiency of individual end-use sectors and of the whole energy system. The application of the principle should also support investments in energy-efficient solutions contributing to the environmental objectives of Regulation (EU) 2020/852 of the European Parliament and of the Council (6).
'''

'''
Footnotepart of the Document 32023L1791...:
(6)  Regulation (EU) 2020/852 of the European Parliament and of the Council of 18 June 2020 on the establishment of a framework to facilitate sustainable investment, and amending Regulation (EU) 2019/2088 (OJ L 198, 22.6.2020, p. 13).
'''

expected_text_result = "Consequently, the energy efficiency first principle should help increase the efficiency of individual end-use sectors and of the whole energy system. The application of the principle should also support investments in energy-efficient solutions contributing to the environmental objectives of Regulation (EU) 2020/852 of the European Parliament and of the Council (Regulation (EU) 2020/852 of the European Parliament and of the Council of 18 June 2020 on the establishment of a framework to facilitate sustainable investment, and amending Regulation (EU) 2019/2088 (OJ L 198, 22.6.2020, p. 13).)."

TEXT_HTML = """<p class="oj-normal">Consequently, the energy efficiency first principle should help increase the efficiency of individual end-use sectors and of the whole energy system. The application of the principle should also support investments in energy-efficient solutions contributing to the environmental objectives of Regulation (EU)&nbsp;2020/852 of the European Parliament and of the&nbsp;Council&nbsp;<a id="ntc6-L_2023231EN.01000101-E0006" href="#ntr6-L_2023231EN.01000101-E0006">(<span class="oj-super oj-note-tag">6</span>)</a>.</p> """

FOOTNOTE_TEXT_HTML = """<p class="oj-note"><a id="ntr6-L_2023231EN.01000101-E0006" href="#ntc6-L_2023231EN.01000101-E0006">(<span class="oj-super">6</span>)</a>&nbsp;&nbsp;Regulation (EU)&nbsp;2020/852 of the European Parliament and of the Council of 18&nbsp;June 2020 on the establishment of a framework to facilitate sustainable investment, and amending Regulation (EU)&nbsp;2019/2088 (<a href="./../../../legal-content/EN/AUTO/?uri=OJ:L:2020:198:TOC">OJ&nbsp;L&nbsp;198, 22.6.2020, p.&nbsp;13</a>).</p>"""

document_dict = {'32023L1791.json': {"html": TEXT_HTML + FOOTNOTE_TEXT_HTML}}

preprocessor = PreprocessorFactory.get_preprocessor(
    preprocessor_name="transform_footnotes_preprocessor"
)

result_dict = preprocessor.preprocess(document_dict, False)
soup = BeautifulSoup(result_dict['32023L1791.json']['html'], 'html.parser')


def test_removing_footnote() -> None:
    paragraphs = soup.find_all('p', class_='oj-note')
    assert (len(paragraphs) == 0)


def test_replacing() -> None:
    result_text = soup.get_text()

    # For correct text-comparing, ignore the HMTL tag &nbsp; (\xa0 (non-breaking space))
    result_text = result_text.replace(u'\xa0', u' ')

    assert (expected_text_result in result_text)
