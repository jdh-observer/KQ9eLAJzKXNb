# General Notes

* .ipynb is the canonical copy
* It was synced with the submitted repository and then the changes were made
* My scripts should go in scripts not script folder. The script folder is there due to being junk they have
* The editor is saying that they added a new markdown cell after the figure. Another option is that the comment can sit in the same cell as the figure/code. It is fine that it is afterwards.
* I have uploaded the scripts
* There are two folders. One is unverified.bib and another is verified.bib. I need to manually add the unverified.bib ones. 


Dear Nabeel,

Thank you for your submission. Some initial reviews have been carried out before sending your article for peer review, in order to ensure it fits our guidelines.


As you can see here, some formatting changes regarding labelling and anchoring have been made.
Please note that if any changes are made to your submission, this is the point to restart from: https://github.com/jdh-observer/KQ9eLAJzKXNb/, as several additions have been made.


Here are just 3 points to consider:
You mention scripts/captioning_script.ipynb, but the .ipynb file is not part of the submitted repository — was this forgotten?
Regarding the labelling, we weren't sure whether you would like to have a separate cell for commenting on the image; see the choice made below.

Lastly, you didn't follow our recommendation to use Zotero and insert citations using the plugin — did you encounter any issues with this?
The article can proceed to peer review with plain text citations and bibliography, but these points should be resolved by the design phase.

Thanks in advance for your feedback.

Elisabeth

---

## Reply from Nabeel (drafted 23 July 2026)

Dear Elisabeth,

Thank you for the careful initial review and for reformatting the notebook. I have restarted from your canonical repository (jdh-observer/KQ9eLAJzKXNb) and worked from there, as you advised. Below is a summary of the changes I have made in response to your three points, together with a few content corrections I identified while preparing the resubmission.

**1. Missing scripts.** The two scripts referenced in the notebook, `captioning_script.ipynb` and `topic_modeling_script.ipynb`, were genuine working pipelines that had simply never been committed to the submitted repository. I have now added both to `scripts/` in the editor's repository, so the commented references in cells 13 and 16 resolve correctly.

**2. Labelling and comment cells.** I am happy to adopt the pattern of a separate italic comment cell per figure, anchor-linked to the figure tag. While implementing it, I noticed and corrected two small issues. First, the anchors in the comment cells for Figures 9 and 10 used uppercase `#FIGURE-` whilst the corresponding figure tags are lowercase `#figure-`, so those two links did not resolve; I have lowercased both. Second, the comment cells began with `* `, which renders as a bullet rather than italic emphasis; I have reformatted all ten comment cells so the emphasis now renders as intended.

**3. Zotero and the citation plugin.** No blocking issue prevented me from adopting the plugin; I simply did not use it for the initial submission. I will migrate to the Zotero citation-manager workflow before resubmitting, re-inserting each in-text citation against my Zotero library and regenerating the bibliography. This will also reconcile the current bibliography, removing orphan references that lack in-text citations and aligning the `.bib` file with the visible reference list.

**Additional content corrections.** Whilst preparing the resubmission I identified and corrected four factual issues in the article itself. First, the gender-term regex in the analysis cell included `him` in the male-term list, which inflated the male count and produced a ratio of 3.92:1, contradicting the 3.69:1 figure reported in the prose; I have removed `him`, and the cell now reproduces 3.69:1 (101,735 / 27,583) exactly. Second, the article stated that "the only photograph connected to Howard University" shows President Coolidge at a graduation; in fact there are twelve such photographs (four of Coolidge at a 1924 graduation, plus eight of classrooms, laboratories, a workshop, and the Law Department building), and I have rewritten the sentence accordingly, preserving the original argument that the institution is framed through outside observers rather than its own intellectual life. Third and fourth, the KeyBERT reference was cited as a 2020 paper when its DOI resolves to a 2021 Zenodo software release; I have recharacterised it as [Computer software] with the year 2021 in both the reference list and the in-text citation.

I would be grateful if you could confirm whether these changes are in order before I proceed with the Zotero migration and the final resubmission. I am happy to make any further adjustments you advise.

Many thanks again for the thorough work on the notebook.

Best,
Nabeel