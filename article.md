---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region tags=["title"] -->
# Captioning the Capital: Multimodal Topic Modelling and the National Photo Company Collection
<!-- #endregion -->

<!-- #region tags=["contributor"] -->
### Nabeel Siddiqui [![orcid](https://orcid.org/sites/default/files/images/orcid_16x16.png)](https://orcid.org/0000-0002-6126-5833) 
Susquehanna University
<!-- #endregion -->

<!-- #region tags=["copyright"] -->
[![cc-by](https://licensebuttons.net/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/) 
© Nabeel Siddiqui. Published by De Gruyter in cooperation with the University of Luxembourg Centre for Contemporary and Digital History. This is an Open Access article distributed under the terms of the [Creative Commons Attribution License CC-BY](https://creativecommons.org/licenses/by/4.0/)

<!-- #endregion -->

<!-- #region tags=["abstract"] -->
This article uses vision-language models and neural topic modelling to analyse 35,368 digitised photographs from the National Photo Company, a photography agency that operated in Washington, D.C., during the first half of the twentieth century and whose collections are housed at the Library of Congress. By generating natural language captions with Florence-2 and clustering them with BERTopic, the study identifies 527 distinct subject categories dominated by government offices and military formations. In addition, a nearly four-to-one ratio of male to female references in the captions quantifies the collection's systematic marginalisation of women. The 23.7 per cent of images that resist topical classification (the noise cluster) reveals both the limits of automated analysis and unexpected archival heterogeneity. Overall, these findings suggest that commercial photography agencies like the National Photo Company operated as a "visual infrastructure of state legitimation," reinforcing governmental authority through standardised compositional conventions.
<!-- #endregion -->

<!-- #region tags=["keywords"] -->
distant viewing, topic modelling, visual culture, digital humanities, computer vision, BERTopic, Florence-2, Library of Congress, Progressive Era, photojournalism
<!-- #endregion -->

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import re
from pathlib import Path
from IPython.display import display, Image, HTML

# Paths
RESULTS_DIR = Path("scripts/results")
FIGURES_DIR = Path("figures")
```

<!-- #region citation-manager={"citations": {"0b6p9": [{"id": "17970137/CCPYPJNV", "source": "zotero"}], "1v8ks": [{"id": "17970137/799URSTD", "source": "zotero"}], "211bc": [{"id": "17970137/4E7L8EWK", "source": "zotero"}], "283fe": [{"id": "17970137/6BIEITXL", "source": "zotero"}], "3fbix": [{"id": "17970137/24TN4JIG", "source": "zotero"}], "8r928": [{"id": "17970137/H5IBYMNW", "source": "zotero"}], "ask09": [{"id": "17970137/2DPNW9Q8", "source": "zotero"}], "dcopw": [{"id": "17970137/X3Q5JQUX", "source": "zotero"}], "fu1sv": [{"id": "17970137/799URSTD", "source": "zotero"}], "ui389": [{"id": "17970137/HBWAKTCH", "source": "zotero"}], "wutgv": [{"id": "17970137/LAQHGXKA", "source": "zotero"}], "xbzb2": [{"id": "17970137/799URSTD", "source": "zotero"}], "y3m8u": [{"id": "17970137/DSFJSTTI", "source": "zotero"}], "zrd3q": [{"id": "17970137/BVRKVU5J", "source": "zotero"}]}} -->
## Introduction

In 1912, Herbert E. French acquired the National Photo Company, a commercial photography agency headquartered in Washington, D.C. <cite id="xbzb2"><a href="#zotero%7C17970137%2F799URSTD">(Library of Congress, n.d.)</a></cite>. From his Washington office, French instructed his photographers to focus on subjects he felt competing agencies had neglected <cite id="ask09"><a href="#zotero%7C17970137%2F2DPNW9Q8">(Library of Congress Research Guides, n.d.)</a></cite>. This strategic orientation gained him entry to government buildings and events, allowing his photographers to document the daily functions of federal institutions. By the 1920s, the National Photo Company had established itself as a leading commercial photography firm in the area and provided images to major newspapers and government publications throughout the Progressive Era and into the early years of the Great Depression.

Many of the images produced by the National Photo Company captured defining moments in the city's physical and symbolic transformation. As Constance McLaughlin Green highlights, during the agency's operating years, the city began shifting from its earlier identity as a quiet Southern town into the bustling seat of national government <cite id="zrd3q"><a href="#zotero%7C17970137%2FBVRKVU5J">(Green, 1963)</a></cite>. The McMillan Plan of 1901-1902 provided concrete shape to this change, seeking to restore Pierre L'Enfant's original vision through neoclassical architecture and monumental scale <cite id="dcopw"><a href="#zotero%7C17970137%2FX3Q5JQUX">(Reps, 1967)</a></cite>. Structures like Union Station, the Lincoln Memorial, and the reconfigured National Mall projected authority through architectural grandeur. As this new infrastructure sought to organise physical space to communicate governmental power, the National Photo Company similarly arranged visual space using repeated compositional formulas that reinforced similar messages.

The economic upheaval of the 1930s brought an abrupt end to the National Photo Company's two-decade visual chronicling of the nation's capital. The Great Depression gutted the newspaper industry, and by the early 1930s French was forced to close the studio's final location. However, French himself proved more resilient than his enterprise. For nearly two more decades, he kept working as an independent commercial photographer, operating from his Southeast Washington home until his death. 

In 1947, the Library of Congress acquired approximately 80,000 photographic prints and glass negatives from French, forming a collection that "documents virtually all aspects of Washington, D. C., life" spanning the years 1909 to 1932 <cite id="1v8ks"><a href="#zotero%7C17970137%2F799URSTD">(Library of Congress, n.d.)</a></cite>. Of these, around thirty-five thousand images have been digitised and made accessible to the public through the Library's Prints and Photographs Online Catalog (PPOC) <cite id="fu1sv"><a href="#zotero%7C17970137%2F799URSTD">(Library of Congress, n.d.)</a></cite>. Navigating these images or analysing them for subject coherence to understand Progressive Era Washington presents significant challenges because of the collection's limited metadata. Aside from brief descriptions in each filename, photos lack organised information on figures, dates, locations, or production contexts. As Barbara Orbach Natanson observes, with nearly 14 million images within its holdings, the Library's staff "are rarely able to undertake in-depth research on individual images" <cite id="283fe"><a href="#zotero%7C17970137%2F6BIEITXL">(Natanson, 2007)</a></cite>. Addressing these challenges at scale requires developing new automated methods, but endeavours to employ algorithmic methods to understand this collection and others like it have encountered notable limitations.

Prior algorithmic work on this archive detected specific image features but has failed to detail broader subject relationships. From May 2021 to January 2022, Lauren Tilton and Taylor Arnold led the Access and Discovery of Documentary Images (ADDI) project at the Library of Congress Labs <cite id="3fbix"><a href="#zotero%7C17970137%2F24TN4JIG">(Arnold &#38; Tilton, 2022)</a></cite>. This initiative used computer vision algorithms on five documentary photography collections, totalling about 315,000 items, including the National Photo Company Collection. It sought to extract discrete attributes like the presence of faces, recognised objects, and poses. Thus, an image might be tagged as containing two people standing outdoors, for instance, without any indication of who they are or why they appear together. Although these are laudable initial steps, the approach has limited capacity to interpret the unique visual logics that confer meaning on pictures because it focuses on low-level features.

The disparity between extracting discrete attributes and interpreting meaning underscores a broader issue with algorithmic approaches to visual culture. Scholars of visual culture have shown that photographs do more than passively record reality. Instead, they actively shape perceptions through selective framing and subject choice <cite id="y3m8u"><a href="#zotero%7C17970137%2FDSFJSTTI">(Tagg, 1988)</a></cite>, <cite id="ui389"><a href="#zotero%7C17970137%2FHBWAKTCH">(Sekula, 1986)</a></cite>,  <cite id="0b6p9"><a href="#zotero%7C17970137%2FCCPYPJNV">(Sontag, 1990)</a></cite>. If photographs are structured by such choices, the analytical task becomes one of identifying the conventions that govern them. Stuart Hall offers a useful conceptual tool for this work, with his notion of "regimes of representation," the historically specific systems through which visual difference is produced and made meaningful <cite id="wutgv"><a href="#zotero%7C17970137%2FLAQHGXKA">(Hall, 1997)</a></cite>. This framework directs attention not only to image content but also to the social conventions governing photographic practice, such as the tacit rules determining who is photographed and in what contexts they appear.

This article investigates whether algorithmic analysis can elucidate such regimes in the National Photo Company collection. Are specific subjects consistently presented to reinforce authority? Are these tendencies influenced by gender or racial factors? Are certain subjects depicted in a formal or informal manner, individually or in groups? Rather than relying on traditional computer vision techniques that focus on object detection, this study employs recent developments in vision-language models to generate natural language captions for each image. These captions record not only what is visible but also how it is contextualised, such as whether a figure is seated behind a desk or standing at a podium, or whether a group is characterised as an assembly or a delegation.

More specifically, this article pairs Florence-2, a vision-language model, with BERTopic for neural topic modelling. This pairing is deliberate. Although BERTopic can cluster images directly through their CLIP embeddings <cite id="211bc"><a href="#zotero%7C17970137%2F4E7L8EWK">(Radford et al., 2021)</a></cite>, this study models the captions instead. That choice follows Arnold and Tilton, who show that turning images into language descriptions can support explainable search, clustering, and recommendation across a photographic collection while avoiding the pitfalls of working directly from visual embeddings <cite id="8r928"><a href="#zotero%7C17970137%2FH5IBYMNW">(Arnold &#38; Tilton, 2024)</a></cite>. Whereas they build an interactive interface for exploring and retrieving individual images, this study applies the same principle to a different use, using neural topic modelling on the captions to surface the subject categories that organise the archive as a whole. BERTopic represents each caption as a point in a semantic space and groups captions with similar wording into the same topic. For instance, if hundreds of captions mention "men in suits seated at a conference table," the model places them in one topic, and the photographs whose captions lie closest to that topic's centroid serve as its most representative examples, which a scholar can examine to judge whether the grouping reflects a coherent subject, such as government hearings.

The pipeline presented herein introduces several key innovations. It employs topic modelling to identify coherent groups across the entire archive, thus unveiling relationships beyond simple pairwise image similarities. By determining which caption embeddings are nearest to each cluster's centroid, it highlights exemplary photographs for scholars to scrutinise as visual evidence of the groupings. It explicitly acknowledges and documents images that defy categorisation, designated as the noise cluster, thereby treating methodological uncertainty as a form of discovery rather than concealing it. When applied to the National Photo Company Collection, this pipeline discerned 527 distinct topics predominantly related to government offices, military formations, and masculine leisure activities, showing how French's agency functioned as a visual infrastructure of state legitimation.
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"19osp": [{"id": "17970137/9CT6669J", "source": "zotero"}], "314hq": [{"id": "17970137/3I47GI3L", "source": "zotero"}], "3l4nu": [{"id": "17970137/YFVNDSCE", "source": "zotero"}], "4ibu3": [{"id": "17970137/YFVNDSCE", "source": "zotero"}], "59iip": [{"id": "17970137/V6LU8JJE", "source": "zotero"}], "amttw": [{"id": "17970137/ZXJ4FHK6", "source": "zotero"}], "blgxg": [{"id": "17970137/TK7HWQH7", "source": "zotero"}], "cmgf6": [{"id": "17970137/ZKGMP49X", "source": "zotero"}], "i3vpn": [{"id": "17970137/YFVNDSCE", "source": "zotero"}], "kgjec": [{"id": "17970137/V6LU8JJE", "source": "zotero"}], "n1p6q": [{"id": "17970137/V6LU8JJE", "source": "zotero"}], "ng4ok": [{"id": "17970137/KHSFXVNA", "source": "zotero"}], "pxhrg": [{"id": "17970137/9SF4LTTE", "source": "zotero"}], "uejay": [{"id": "17970137/CJ7LNUTM", "source": "zotero"}], "yvya6": [{"id": "17970137/2SRZSPH2", "source": "zotero"}]}} -->
## Navigating the Visual Turn in Digital History

The methodological and theoretical landscape of this study exists at the intersection of several disciplines, including distant viewing within the digital humanities, neural topic modelling, and the critical analysis of visual culture. Each perspective provides valuable insights into the challenges of analysing extensive photographic corpora, and their integration creates new opportunities for automated cultural research.

### The Visual Turn and the "Laocoön Problem" of Computation

During its early development, digital humanities scholarship primarily concentrated on textual analysis. Techniques such as "distant reading" emerged to address the vast expanse of printed material that exceeds any individual's capacity for close examination. Margaret Cohen famously termed this phenomenon the "great unread" <cite id="amttw"><a href="#zotero%7C17970137%2FZXJ4FHK6">(Cohen, 1999)</a></cite>. However, as Melvin Wevers and Thomas Smits argue, the discipline is poised for a "visual digital turn," enabled by the growing availability of digitised visual archives and the development of deep neural networks capable of classifying images at scale <cite id="yvya6"><a href="#zotero%7C17970137%2F2SRZSPH2">(Wevers &#38; Smits, 2020)</a></cite>.

The shift from text to image analysis involves more than methodological adjustments. Visual digital humanities confronts what Leonardo Impett and Fabian Offert term the "Laocoön problem of computation" <cite id="kgjec"><a href="#zotero%7C17970137%2FV6LU8JJE">(Impett &#38; Offert, 2024)</a></cite>. Extending Gotthold Ephraim Lessing's eighteenth-century division between spatial arts (painting) and temporal arts (poetry), Impett and Offert contend that images and text have divergent affordances that are "almost diametrically opposed in the digital realm" <cite id="n1p6q"><a href="#zotero%7C17970137%2FV6LU8JJE">(Impett &#38; Offert, 2024)</a></cite>. Unlike text, which arrives pre-segmented into characters, words, and sentences with clear hierarchical organisation, image data remains inherently continuous. Digital images comprise pixel grids without natural boundaries or units. "Where do image-objects end, exactly?" Impett and Offert ask <cite id="59iip"><a href="#zotero%7C17970137%2FV6LU8JJE">(Impett &#38; Offert, 2024)</a></cite>. This lack of discrete elements has traditionally constrained computational analysis of visual materials to extracting low-level features. 

Such features, however, register form rather than meaning. Effective at detecting stylistic similarities between images, these early extraction methods could not interpret the data's semantic content and often failed to grasp what a picture actually depicted. This challenge, termed the "semantic gap" by <cite id="blgxg"><a href="#zotero%7C17970137%2FTK7HWQH7">(Smeulders et al., 2000)</a></cite>, describes the mismatch between the information that can be extracted from visual data and the interpretation that the same data have for a user in a given situation. Nanne van Noord (2022) emphasises that this gap is particularly salient for culturally significant images like iconic photographs, where much of their meaning lies in aspects "not being described by [their] visual data" <cite id="pxhrg"><a href="#zotero%7C17970137%2F9SF4LTTE">(van Noord, 2022)</a></cite>. This semantic gap becomes especially acute when computer vision models confront historical archives, whose visual conventions differ sharply from the material on which such models are trained.

Models trained on contemporary data frequently produce problematic outcomes when applied to historical archives. Collections containing millions of images, such as ImageNet <cite id="uejay"><a href="#zotero%7C17970137%2FCJ7LNUTM">(Russakovsky et al., 2015)</a></cite> or COCO <cite id="ng4ok"><a href="#zotero%7C17970137%2FKHSFXVNA">(Lin et al., 2014)</a></cite>, mirror the visual conventions of the twenty-first-century internet. When applied to historical materials, these models produce anachronistic or nonsensical outputs. For example, a model trained on recent data may categorise a Progressive Era hat as a "bucket" or a historical streetcar as a "bus," owing to the labels in its training dataset.

Beyond temporal mismatch, Kate Crawford and Trevor Paglen have illuminated the deeper political implications embedded within training datasets themselves <cite id="314hq"><a href="#zotero%7C17970137%2F3I47GI3L">(Crawford &#38; Paglen, 2021)</a></cite>. In their analysis of ImageNet's "Person" category, for instance, they identify labels such as "failure," "loser," and "alcoholic" assigned to ordinary photographs of individuals. Under the pretence of neutral categorisation, they reveal how classificatory systems incorporate cultural biases. Thus, training data is not merely raw material but constitutes a political intervention, prompting urgent inquiries into who determines the existence of categories and how images are allocated to them. As Lev Manovich asserts, decisions about "what is chosen as objects, what features are chosen, and how these features are encoded" are not neutral; they determine how cultural phenomena become "computable, manageable and knowable" through computational analysis <cite id="cmgf6"><a href="#zotero%7C17970137%2FZKGMP49X">(Manovich, 2015)</a></cite>.

To address the interpretive challenges of the visual turn, Taylor Arnold and Lauren Tilton introduced the concept of "distant viewing," modelled after Franco Moretti's "distant reading" <cite id="4ibu3"><a href="#zotero%7C17970137%2FYFVNDSCE">(Arnold &#38; Tilton, 2019)</a></cite>. In "Conjectures on World Literature," Moretti suggested that stepping back from individual texts to analyse patterns across genres and national literatures could reveal structures hidden from close reading. He called this approach "distant reading," where "distance... is a condition of knowledge" <cite id="19osp"><a href="#zotero%7C17970137%2F9CT6669J">(Moretti, 2000)</a></cite>. Arnold and Tilton adapt this scalar shift, introducing distant viewing as a framework "distinguished from other approaches by making explicit the interpretive nature of extracting semantic metadata from images" <cite id="i3vpn"><a href="#zotero%7C17970137%2FYFVNDSCE">(Arnold &#38; Tilton, 2019)</a></cite>. The core idea of the distant viewing framework is that visual metadata is not simply "given" but actively constructed. Unlike a text corpus, where data is already described by its characters and syntax, a visual corpus requires the researcher to develop "a code system in semiotics or, similarly, a metadata schema in informatics" <cite id="3l4nu"><a href="#zotero%7C17970137%2FYFVNDSCE">(Arnold &#38; Tilton, 2019)</a></cite>. For them, this process unfolds in a series of steps: extracting semantic features with computer vision, aggregating the extracted metadata across the corpus, and visualising or analysing the resulting patterns through exploratory data analysis. This sequence has been applied successfully to many collections. Yet because every step depends on features that a model can reliably detect, the framework also reveals the limits of reducing an image to discrete, detectable elements.
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"3muoc": [{"id": "17970137/IYV2WM3S", "source": "zotero"}], "ccij3": [{"id": "17970137/LS4UG6JJ", "source": "zotero"}], "ihggk": [{"id": "17970137/YI6YHSEP", "source": "zotero"}], "itwzl": [{"id": "17970137/FU3ZC6ZL", "source": "zotero"}], "sysha": [{"id": "17970137/F4X7LWSN", "source": "zotero"}], "yod4x": [{"id": "17970137/VZ7HGW5B", "source": "zotero"}]}} -->
### From Bags of Words to Neural Embeddings

If computer vision provides the tools for "viewing" the archive, then topic modelling offers the framework for "reading" its topical structure. Since its introduction, topic modelling has become a key method in the digital humanities. Elijah Meeks and Scott B. Weingart describe it as a "synecdoche" for the field itself, a single approach that highlights the broader shift towards machine-assisted discovery <cite id="3muoc"><a href="#zotero%7C17970137%2FIYV2WM3S">(Meeks &#38; Weingart, 2012)</a></cite>. Traditional topic modelling relies on Latent Dirichlet Allocation (LDA) <cite id="itwzl"><a href="#zotero%7C17970137%2FFU3ZC6ZL">(Blei et al., 2003)</a></cite>, a probabilistic model that treats each document as a mixture of underlying topics. However, LDA has notable limits when used with short texts. Its "bag-of-words" method treats each document as a collection of independent words, ignoring word order and contextual meanings essential to human communication. For short texts, where each word can be highly important, this limitation can result in poor topic coherence and lower interpretability. Because of these difficulties, researchers have sought new methods that help better capture meaning. One particularly promising approach is neural topic modelling.

Maarten Grootendorst's BERTopic best exemplifies neural topic modelling <cite id="ihggk"><a href="#zotero%7C17970137%2FYI6YHSEP">(Grootendorst, 2022)</a></cite>. Unlike LDA, which relies on frequency counts, BERTopic uses transformer-based embeddings with BERT and its variants <cite id="yod4x"><a href="#zotero%7C17970137%2FVZ7HGW5B">(Devlin et al., 2019)</a></cite>. These models represent words and documents as high-dimensional vectors that capture rich contextual information. In the BERTopic process, documents are embedded and dimensionality-reduced before being clustered using density-based algorithms like HDBSCAN. This approach allows for the identification of coherent subject clusters even within very short or linguistically sparse texts.

The integration of computer vision and neural topic modelling provides a compelling methodology for the large-scale analysis of image archives. As Julia Thomas and Irene Testini assert, this methodology enables searching the content of illustrations in large datasets <cite id="sysha"><a href="#zotero%7C17970137%2FF4X7LWSN">(Thomas &#38; Testini, 2024)</a></cite>. William J. Mitchell offers a theoretical justification for this approach, noting that "there are no visual media" because "all media are mixed media," involving multiple sensory modalities and semiotic systems <cite id="ccij3"><a href="#zotero%7C17970137%2FLS4UG6JJ">(Mitchell, 2005)</a></cite>. In short, this "multimodal turn" in topic modelling allows researchers to transcend the artificial separation between pictorial and textual discourse. By translating pictorial evidence into natural language and then clustering it through neural embeddings, scholars can identify recurring thematic patterns and subject categories that structure archival collections.
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"1k6id": [{"id": "17970137/GEYL3A6G", "source": "zotero"}], "3sgpg": [{"id": "17970137/LAQHGXKA", "source": "zotero"}], "gb7ej": [{"id": "17970137/N5YY7JRY", "source": "zotero"}], "mhicp": [{"id": "17970137/DSFJSTTI", "source": "zotero"}], "rn1aw": [{"id": "17970137/MHHLIP9P", "source": "zotero"}], "ryet7": [{"id": "17970137/LAQHGXKA", "source": "zotero"}]}} -->
### Visual Culture and the Politics of the Archive

Trends identified through topic modelling are not inherently self-explanatory. Instead, they necessitate a theoretical framework that conceptualises photography as a sphere of social influence. Scholars in visual culture have articulated a comprehensive set of interconnected concepts that detail the content of representation and the social conventions guiding photographic practice. Collectively, these serve as interpretative tools for this study. These scholars share a fundamental premise. Photographic archives do not merely record reality passively but actively shape it through systematic processes of inclusion and exclusion.

Topic modelling surfaces recurring groupings of subjects, but these clusters do not interpret themselves. To read them as evidence of how an archive represents its world, this study turns to Stuart Hall's concept of "regimes of representation" <cite id="3sgpg"><a href="#zotero%7C17970137%2FLAQHGXKA">(Hall, 1997)</a></cite>. For Hall, representation does not merely reflect pre-existing meanings but actively produces them through discourse. He theorised regimes of representation as "the whole repertoire of imagery and visual effects through which 'difference' is represented at any one historical moment" <cite id="ryet7"><a href="#zotero%7C17970137%2FLAQHGXKA">(Hall, 1997)</a></cite>. These are systems of meaning-production that operate through relations of power/knowledge to construct and maintain categories of difference. When applied to photographic archives, Hall's framework enables scholars to interpret topic clusters not merely as neutral categories but as manifestations of historically specific regimes that privileged certain subjects and marginalised others. While Hall specifies *what* representations achieve through discourse, Pierre Bourdieu elucidates *why* certain subjects are initially selected for photographic depiction.

Pierre Bourdieu's analysis of photography as a "middle-brow art" elucidates the social logic underlying Hall's regimes <cite id="gb7ej"><a href="#zotero%7C17970137%2FN5YY7JRY">(Bourdieu, 1990)</a></cite>. Bourdieu argues that photographic practice is governed by implicit conventions defining what is considered worthy of being photographed, conventions that create and reinforce hierarchies of legitimacy by distinguishing high culture from low and valued subjects from disregarded ones. Building on Bourdieu, Marco Solaroli shows that in commercial news photography these conventions function as the "rules of the field," governing production practices and symbolic struggles for distinction within the profession <cite id="1k6id"><a href="#zotero%7C17970137%2FGEYL3A6G">(Solaroli, 2016)</a></cite>. Beyond such social expectations, moreover, the photographic system itself constrains what can be depicted.

Vilém Flusser and John Tagg expand the analysis from social conventions to the material and institutional systems of photography <cite id="rn1aw"><a href="#zotero%7C17970137%2FMHHLIP9P">(Flusser, 2000)</a></cite>, <cite id="mhicp"><a href="#zotero%7C17970137%2FDSFJSTTI">(Tagg, 1988)</a></cite>. Flusser asserts that technical images are produced by apparatuses that are "programmed" to realise a restricted set of possibilities. Thus, the camera's internal logic defines the range of photographs that can exist. Tagg further broadens this notion to include institutions by arguing that photography serves as a tool of power and surveillance that reinforces the authority of the state. Thus, for Tagg, the meaning of a photograph cannot be separated from the institutional contexts of its production and circulation. Collections such as police archives, medical records, identity documents, and news services all deploy photography to make populations visible in ways that serve administrative and disciplinary ends.

These theoretical insights establish a framework for interpreting the results of computational analysis without overstating what such analysis can achieve. Algorithms can identify patterns, but they cannot explain why those patterns exist. Explanation requires historical reasoning informed by the understanding that photographic practice is shaped by technological constraints and social conventions about who merited public visibility. The topic modelling pipeline described in the following section is designed with these limitations in mind: it surfaces patterns that invite interpretation rather than conclusions that foreclose it.
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"4yyg4": [{"id": "17970137/KL9FD2CB", "source": "zotero"}], "zjzce": [{"id": "17970137/Q3CB4DLN", "source": "zotero"}]}} -->
## Building the Pipeline: Data and Methods

The theoretical and methodological frameworks outlined above guide the development of this study's analytical pipeline. It includes several key components, such as gathering data from archival sources, generating captions using vision-language models, and applying neural topic modelling to categorise subjects. This section explains each stage, emphasising decisions that influence later analysis.

### Assembling the Corpus

This project obtained the National Photo Company Collection from Wikimedia Commons rather than directly from the Library of Congress's original repository <cite id="zjzce"><a href="#zotero%7C17970137%2FQ3CB4DLN">(Wikimedia Commons, n.d.)</a></cite>. Although the Library of Congress hosts the digitised collection, Wikimedia Commons, which replicates these public domain images, provides more permissive rate limits that enable large-scale downloads. This reliability was essential for retrieving all images (n = 35,368) despite occasional network disruptions. The download was finalised on 24 December 2024, and was subsequently filtered for commonly used image formats (jpg, png, gif) to ensure compatibility with subsequent processing. Utilising Wikimedia Commons as the source does not alter the images, which remain identical to those maintained by the Library of Congress.

The resulting high-resolution images required compression to optimise efficiency, as the aggregate size exceeded 93 gigabytes. Utilising this volume at full archival quality was impractical due to memory limitations and lengthy processing durations. To address this issue, ImageMagick was used to resize all images to a maximum width of 800 pixels while preserving their aspect ratio <cite id="4yyg4"><a href="#zotero%7C17970137%2FKL9FD2CB">(ImageMagick Studio LLC, 2024)</a></cite>. This adjustment reduced the dataset size by 87 per cent, from 93 gigabytes to approximately 12 gigabytes. Although reducing the resolution inevitably leads to some loss of fine details from the original scans, pilot testing on a sample of images confirmed that the lower resolution still enabled Florence-2 to generate detailed captions. While typographical elements and distant background details became less distinct, the primary compositional arrangements and visual conventions essential to this study's analytical objectives remained clearly discernible.

The 35,368 digitised images make up about 44 per cent of the roughly 80,000 items the Library of Congress acquired in 1947. The method used for selecting items for digitisation is not well documented. Thus, the digital collection might not fully represent the entire archive. This limitation is common in large digital humanities projects, where the digitisation process can create selection biases that affect subsequent automated analysis. Therefore, researchers should interpret results as relating to the digitised subset rather than the whole National Photo Company archive. After assembling the corpus, the next step was to create textual descriptions for each photograph.
<!-- #endregion -->

```python
# ============================================================
# IMAGE DOWNLOAD PIPELINE
# ============================================================
# Images were downloaded using gallery-dl on December 24, 2024
#
# Command used (zsh):
# gallery-dl https://commons.wikimedia.org/wiki/Category:National_Photo_Company_Collection \
#   --filter "extension in ('jpg', 'png', 'gif')"
#
# This downloaded 35,368 images filtered for common image formats
#
# Images were then resized to 800px width using ImageMagick:
# for img in *.jpg *.png *.jpeg; do
#   mogrify -resize 800x "$img"
# done
# ============================================================
```

<!-- #region citation-manager={"citations": {"1lydj": [{"id": "17970137/FAEP32ZA", "source": "zotero"}], "x3sje": [{"id": "17970137/F4X7LWSN", "source": "zotero"}]}} -->
### Generating Captions with Florence-2

After assembling and preprocessing the corpus, the subsequent phase of the workflow involved generating a textual description for each photograph. The aim was to make an image archive searchable and analysable through text, a goal this study shares with recent work such as that of Julia Thomas and Irene Testini. They use AI to detect and isolate the captions already printed alongside historical book illustrations and then read them with optical character recognition, so that the illustrations can be searched by their existing captions <cite id="x3sje"><a href="#zotero%7C17970137%2FF4X7LWSN">(Thomas &#38; Testini, 2024)</a></cite>. The photographs studied here, by contrast, carry no such captions, so rather than recovering existing text, this study uses Florence-2 to *generate* a new description of each image's content. In both cases, the purpose is the same, namely to render visual material as text that established text-mining techniques can analyse. This conversion addresses the "semantic gap" discussed earlier by producing discrete textual units from continuous image data. 

Microsoft's Florence-2 large model was chosen as the primary vision-language engine for caption generation for three reasons pertinent to this investigation <cite id="1lydj"><a href="#zotero%7C17970137%2FFAEP32ZA">(Xiao et al., 2023)</a></cite>. First, its extensive training on 126 million diverse images paired with 5.4 billion text descriptions furnishes a comprehensive visual vocabulary for generating detailed captions. Second, a single text prompt controls the model's behaviour, so one system can perform several vision tasks, such as detailed captioning or object detection, simply by changing the instruction it is given, rather than requiring a separate specialised model for each task. Third, unlike commercial API services, Florence-2 allows local deployment, enabling researchers to process archival materials without transmitting potentially sensitive data to external servers. 

In particular, the caption generation procedure used Florence-2's `<MORE_DETAILED_CAPTION>` task, which produces more comprehensive descriptions than its standard captioning mode. This methodology emphasises thoroughness over brevity, with captions generally spanning two to four paragraphs. The entire corpus was processed sequentially. Each image was loaded, resized to meet the model's input specifications, and passed through Florence-2's encoder-decoder architecture. Deterministic parameters (temperature = 0, without sampling) were used during generation to ensure reproducibility. As a result, the same image processed multiple times will consistently generate identical captions. This deterministic approach distinguishes the method from commercial AI services that may yield variable outputs, thus providing a reliable foundation for scholarly analysis and replication. The final step involved applying neural topic modelling to the generated captions.
<!-- #endregion -->

```python
# ============================================================
# FLORENCE-2 CAPTION GENERATION
# Actual implementation from scripts/captioning_script.ipynb
# ============================================================

# import os
# import torch
# from PIL import Image
# from transformers import AutoProcessor, AutoModelForCausalLM
# import csv
# import pathlib

# # Setup device and precision
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch_dtype = torch.float16

# # Load Florence-2-large model
# model = AutoModelForCausalLM.from_pretrained(
#     "microsoft/Florence-2-large", 
#     torch_dtype=torch_dtype, 
#     trust_remote_code=True
# ).to(device)

# processor = AutoProcessor.from_pretrained(
#     "microsoft/Florence-2-large", 
#     trust_remote_code=True
# )

# # Configuration
# image_dir = "images_transformed"  # Resized images directory
# output_file = "results/captions.csv"
# prompt = "<MORE_DETAILED_CAPTION>"

# # Process images and generate captions
# # This code was run once to generate captions for all 35,368 images
# # Results are loaded from CSV in the next cell
```

```python
# Load pre-computed captions
captions_df = pd.read_csv(RESULTS_DIR / "captions.csv")
print(f"Loaded {len(captions_df):,} captions")
print(f"\nSample caption (first 300 chars):")
print(captions_df.iloc[0]["caption"][:300] + "...")
```

<!-- #region citation-manager={"citations": {"3i12b": [{"id": "17970137/YXL9CHS2", "source": "zotero"}], "7f8ev": [{"id": "17970137/YI6YHSEP", "source": "zotero"}], "9l1di": [{"id": "17970137/GX7MI9NR", "source": "zotero"}], "sge34": [{"id": "17970137/7G4M5KXE", "source": "zotero"}]}} -->
### Clustering with BERTopic

The final phase of the pipeline involved applying neural topic modelling to the 35,368 captions generated by Florence-2. BERTopic was chosen because, unlike traditional LDA, it generates dense embeddings that capture contextual significance before clustering <cite id="7f8ev"><a href="#zotero%7C17970137%2FYI6YHSEP">(Grootendorst, 2022)</a></cite>. This methodology is especially beneficial for image captions, which, as mentioned earlier, are characteristically more concise than the longer texts typically handled by conventional topic models.

The modular design of BERTopic facilitates customisation at different stages of the topic modelling process. For this investigation, the pipeline was configured as follows: First, documents were transformed into dense numerical embeddings employing the `all-MiniLM-L6-v2` sentence-transformers model <cite id="3i12b"><a href="#zotero%7C17970137%2FYXL9CHS2">(Wang et al., 2021)</a></cite>. Second, these embeddings underwent dimensionality reduction via UMAP utilising default parameters. Third, the reduced embeddings were clustered with HDBSCAN, using BERTopic's default minimum cluster size of 10, which defines the minimum number of documents required to constitute a distinct topic. Fourth, documents within each cluster were amalgamated into a single representative text. Fifth, a class-based TF-IDF calculation (c-TF-IDF) incorporating BM25 weighting, a term frequency normalisation method that reduces the impact of highly common words, was applied to identify characteristic words for each topic <cite id="sge34"><a href="#zotero%7C17970137%2F7G4M5KXE">(Robertson &#38; Zaragoza, 2009)</a></cite>. We then used representation models to clarify topic descriptions. 

Given its significance for the current investigation, the sixth step (topic representation) merits further elaboration. Several complementary representation methods were selected from the options provided by BERTopic. The `KeyBERTInspired` representation, which uses embeddings, was used to identify keywords conceptually proximate to the topic centroid, capturing synonyms that frequency-based approaches might overlook <cite id="9l1di"><a href="#zotero%7C17970137%2FGX7MI9NR">(Grootendorst, 2020)</a></cite>. The `MaximalMarginalRelevance`, with a diversity parameter set at 0.3 (chosen to balance relevance and diversity), was used to ensure the selected keywords are both representative and sufficiently distinct from one another. Most notably, for visual analysis purposes, the `VisualRepresentation` component selected the nine images whose caption embeddings are closest to each topic's centroid—the geometric centre of all embeddings associated with that cluster. This centroid-based selection ensures the representative images best reflect their respective topics, enabling researchers to visually examine what the algorithm has identified as the "core" of each topical cluster.

These configuration choices influence the outcomes of the topic modelling process. Different embedding models, UMAP configurations, or minimum cluster size thresholds may produce alternative topic structures. Consequently, the 527 topics identified represent one informative level of topical detail among various options, and researchers should consider these results as an algorithmically derived interpretation rather than a definitive topical structure.
<!-- #endregion -->

```python
# ============================================================
# BERTOPIC CLUSTERING PIPELINE  
# Actual implementation from scripts/topic_modeling_script.ipynb
# ============================================================

# import os
# from pathlib import Path
# import pandas as pd
# from bertopic import BERTopic
# from bertopic.representation import MaximalMarginalRelevance, KeyBERTInspired, VisualRepresentation
# from bertopic.vectorizers import ClassTfidfTransformer
# from sklearn.feature_extraction.text import CountVectorizer
# from sentence_transformers import SentenceTransformer

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # Load captions and prepare data
# captions_csv = Path("Results/captions.csv")
# captions = pd.read_csv(captions_csv)[['image', 'caption']]

# image_folder_path = "Data"
# image_filenames = captions['image'].tolist()
# docs = captions['caption'].tolist()
# images = [str(Path(image_folder_path) / filename) for filename in image_filenames]

# # Step 1: Load embedding model
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# # Step 2: Define vectorizer and c-TF-IDF model
# vectorizer_model = CountVectorizer(stop_words="english")
# ctfidf_model = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)

# # Step 3: Define representation models (three complementary methods)
# representation_model = {
#     "Visual_Aspect": VisualRepresentation(nr_repr_images=9),  # 9 most representative images
#     "KeyBERT": KeyBERTInspired(),  # Semantically similar keywords
#     "MMR": MaximalMarginalRelevance(diversity=0.3)  # Diverse but relevant keywords
# }

# # Step 4: Build BERTopic model
# topic_model = BERTopic(
#     embedding_model=embedding_model,
#     vectorizer_model=vectorizer_model,
#     ctfidf_model=ctfidf_model,
#     representation_model=representation_model
# )

# # Step 5: Fit model (this code was run once to generate topic assignments)
# # topics, probs = topic_model.fit_transform(documents=docs, images=images)

# # Step 6: Save model and results
# # topic_model.save("my_model", serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)
# # results = pd.DataFrame({"Image": images, "Topic": topics})
# # results.to_csv("Results/topic_model_results.csv", index=False)
```

```python
# Load pre-computed topic assignments
results_df = pd.read_csv(RESULTS_DIR / "topic_model_results.csv")

with open(RESULTS_DIR / "topics.json", "r") as f:
    topics_data = json.load(f)

print(f"Loaded {len(results_df):,} topic assignments")
print(f"Unique topics: {results_df['Topic'].nunique()}")


def get_topic_keywords(topic_id, topics_data):
    """Extract top 5 keywords for a topic."""
    try:
        keywords = topics_data['topic_representations'][str(topic_id)]
        return ", ".join([k[0] for k in keywords[:5]])
    except KeyError:
        return "N/A"
```

<!-- #region tags=["hermeneutics"] -->
## Patterns Emerging from the Archive

The methodology produced consistent patterns within the National Photo Company Collection. This section delineates these findings into four components: an overview of topic distribution, an analysis of primary patterns, case studies of selected topics, and a brief investigation of gender disparities and race.

### The Shape of the Archive

BERTopic identified 527 distinct subject clusters across the 35,368 images. Of these, HDBSCAN classified 8,396 photographs (23.7 per cent) as noise (images that resist coherent topical assignment and are labelled Topic -1). The remaining 26,972 images were organised into 526 meaningful topics. 

Cluster sizes demonstrated considerable variability. Among the 526 meaningful topics, the mean is 51.3 images, while the median is only 32, indicating a right-skewed distribution in which most topics contain relatively few photographs and a minority of larger topics elevate the mean. The smallest meaningful topics comprise 10 images, which was the minimum threshold set, whereas the largest include 1,024 photographs. This skewed distribution reflects the nature of the collection, in which some subjects received consistent photographic attention while others were documented only sporadically.
<!-- #endregion -->

```python jdh={"module": "object", "object": {"source": ["Distribution of Images Across Topics"], "type": "image"}} tags=["figure-topic-distribution-*"]
# Figure 1: Distribution of Images Across Topics

topic_counts = results_df['Topic'].value_counts().sort_index()
topic_counts_no_noise = topic_counts[topic_counts.index != -1]

plt.figure(figsize=(10, 5))
plt.hist(topic_counts_no_noise.values, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
plt.xlabel('Number of Images per Topic')
plt.ylabel('Frequency (Number of Topics)')
plt.title('Distribution of Images Across Topics')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Summary statistics
noise_count = topic_counts.get(-1, 0)
print(f"Total images: {len(results_df):,}")
print(f"Noise cluster (Topic -1): {noise_count:,} ({100*noise_count/len(results_df):.1f}%)")
print(f"Meaningful topics: {len(topic_counts_no_noise)}")
print(f"Mean topic size: {topic_counts_no_noise.mean():.1f}")
print(f"Median topic size: {topic_counts_no_noise.median():.0f}")
```

_[Figure 1](#figure-topic-distribution-*) shows a right-skewed distribution, with most topics containing few photographs and a small number of large topics dominating._


Approximately half of the non-noise images are concentrated across roughly 80 topics, with the top 20 topics encompassing 5,840 images, representing 21.7 per cent of the classified photographs. The subjects that attract the most visual focus (such as government offices, military formations, and formal meetings) align closely with themes of institutional authority and masculinity. This distribution suggests that the National Photo Company did not set out to document life in Washington comprehensively but instead concentrated on a narrow band of subjects. The archive is therefore better understood as a record of commercial and institutional demand than as a neutral survey of the city. Its density around government offices, military formations, and formal meetings indicates that the agency's cameras followed its clients' priorities, favouring the visible machinery of the state over the wider social life of the capital. What the collection makes abundant, in other words, is precisely what newspapers and government publications were most willing to buy.

```python jdh={"module": "object", "object": {"source": ["Cumulative Distribution of Images Across Topics ranked by size"], "type": "image"}} tags=["figure-cumulative-distribution-*"]
# Figure 2: Cumulative distribution

sorted_counts = topic_counts_no_noise.sort_values(ascending=False)
cumulative = np.cumsum(sorted_counts.values)
cumulative_pct = (cumulative / cumulative[-1]) * 100

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(cumulative_pct) + 1), cumulative_pct, linewidth=2, color='darkblue')
plt.xlabel('Number of Topics (Ranked by Size)')
plt.ylabel('Cumulative Percentage of Images (%)')
plt.title('Cumulative Distribution of Images Across Topics')
plt.grid(True, alpha=0.3)
plt.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% of images')
plt.axhline(y=80, color='orange', linestyle='--', alpha=0.5, label='80% of images')
plt.legend()
plt.tight_layout()
plt.show()

# Find concentration
topics_for_50 = np.searchsorted(cumulative_pct, 50) + 1
topics_for_80 = np.searchsorted(cumulative_pct, 80) + 1
print(f"50% of images contained in top {topics_for_50} topics")
print(f"80% of images contained in top {topics_for_80} topics")
```

_The curve in [Figure 2](#figure-cumulative-distribution-*) shows that a small number of large topics make up a disproportionate share of all classified photographs._

The extended range of smaller topics illustrates the collection's diversity beyond institutional authority. Beneath the principal categories, hundreds of more specialised subjects examine particular landmarks, specific types of ceremonies, individual sporting events, or distinctive photographic moments.


### Patterns of Visual Emphasis

An analysis of the content within individual topics reveals distinct tendencies of visual focus. The top ten topics are delineated in [Table 1](#table-1-*), demonstrating consistent themes predominantly centred on governmental authority and civic life.

<!-- #region jdh={"object": {"source": ["Table 1: Top 10 topics by image count with representative keywords"], "type": "table"}} tags=["table-1-*"] -->


| Rank | Topic ID | Image Count | Percentage | Representative Keywords |
|------|----------|-------------|------------|-------------------------|
| 1 | 0 | 1,024 | 2.90% | pen, telephone, supplies, cluttered, writing |
| 2 | 1 | 598 | 1.69% | rifles, tent, rifle, caps, stretcher |
| 3 | 2 | 470 | 1.33% | meeting, conference, discussion, documents, signing |
| 4 | 3 | 353 | 1.00% | glove, action, throwing, pitch, leg |
| 5 | 4 | 337 | 0.95% | younger, older, faces, expressions, cigarette |
| 6 | 5 | 261 | 0.74% | yard, shutters, driveway, chimney, manicured |
| 7 | 6 | 238 | 0.67% | sliding, home, batter, catch, crouched |
| 8 | 7 | 232 | 0.66% | signing, map, document, coming, lockers |
| 9 | 8 | 225 | 0.64% | doormat, apples, younger, glass, argyle |
| 10 | 9 | 222 | 0.63% | pedimented, facade, domes, pediment, levels |
<!-- #endregion -->

Four broad subject categories emerge from these clusters. Firstly, official government activities dominate, encompassing topics related to bureaucratic work (Topic 0), formal meetings (Topic 2), and document signing (Topic 7). Secondly, military subjects (Topic 1) attract considerable attention, reflecting Washington's role in national defence during and after the First World War. Thirdly, leisure and sports, particularly baseball (Topics 3 and 6), depict masculine recreation. Fourthly, architectural and domestic themes (Topics 5 and 9) illustrate Washington's built environment.


### Reading the Visual Conventions

The subsequent case studies analyse four subjects in greater depth: several prominent clusters, including Topics 0, 1, and 2, which garnered the most enduring photographic focus, and Topic 5, which records Washington's built environment. This subject links the collection's human subjects to the physical spaces they occupied. Each case study features the nine photographs whose embeddings are closest to each topic centroid, exemplifying the most representative samples of each theme. Again, the topic modelling algorithm does not see the images after the vision model captions them. 

**Professional Authority (Topic 0)**

The most prominent theme, Topic 0, centres on office environments defined by keywords such as "pen," "telephone," and "supplies." The cluster consists of portraits of officials seated at their desks, where authority derives not from personal charisma or individual identity but from the subject's position amid the tools of administration. Surrounded by the accoutrements of clerical labour, the sitter is presented as the master of the modern state's informational networks, and the bureaucratic workspace becomes a setting for the display of professional expertise. 

This desk portrait functioned as a modular formula that allowed the agency to present a wide range of officials through a single, authoritative lens. The consistency rendered individual bureaucrats interchangeable representatives of a larger institutional machine. Rather than capturing a unique moment, each photograph reinforced a standardised visual grammar that prioritised the stability and continuity of the office over the personality of whoever happened to be seated within it.

This standardisation aligned with the Progressive Era press's preference for an aesthetic of mechanical objectivity where the camera was perceived as a neutral observer of institutional facts rather than a tool of interpretation. By emphasising order and clerical diligence, these images framed the rapid expansion of federal power as the inevitable outcome of professional management.

```python jdh={"module": "object", "object": {"source": ["Representative images from Topic 0 (Professional Authority)"], "type": "image"}} tags=["figure-representative-images-*"]
# Figure 3: Topic 0 representative images
# These composites were pre-generated by BERTopic's VisualRepresentation

display(Image(filename=FIGURES_DIR / "topic_images" / "0.jpg", width=700))
print(f"Topic 0 — {topic_counts.get(0, 'N/A')} images")
print(f"Keywords: {get_topic_keywords(0, topics_data)}")
```

_The nine most representative photographs from this cluster ([Figure 3](#figure-representative-images-*)) illustrate the standardised visual language of desk portraits._

<!-- #region citation-manager={"citations": {"z3s58": [{"id": "17970137/7VSCBJ52", "source": "zotero"}]}} -->
**Military Power (Topic 1)**

The second-largest cluster focuses on military themes, defined by keywords such as "rifles," "tent," "caps," and "stretcher." These images depict personnel engaged in training, ceremonial formations, and official events that reflected Washington's role as the command centre for American military operations during and after the First World War. Like Topic 0, these photographs of military personnel, equipment, and ceremonies participated in the visual legitimation of state power and military authority. The cluster captures a period of unprecedented military expansion in Washington, as the federal government mobilised resources for war and then maintained a significant peacetime military establishment through the 1920s.

The captions generated for Topic 1 reveal a consistent visual vocabulary emphasising collective order over individual action. Image after image describes "men in military uniforms standing in a line," "groups of men in a field," and personnel arranged in "formations" before government buildings. References to uniforms predominate, with caps, helmets, and formal attire appearing repeatedly in the machine-generated descriptions. Rifles appear frequently, but always in contexts of display rather than use, held by soldiers standing at attention or carried during parade formations. This emphasis on the material signifiers of military identity, such as the uniform, the weapon held correctly, and the aligned formation, transformed potentially chaotic scenes of military activity into orderly visual narratives of disciplined power.

Put more simply, this visual consistency worked to naturalise what Michael Griffin, analysing twentieth-century wartime media coverage, terms a "sanitised" version of military imagery, one that emphasised organisational efficiency and preparedness while omitting any suggestion of violence or disorder <cite id="z3s58"><a href="#zotero%7C17970137%2F7VSCBJ52">(Griffin, 2010)</a></cite>. Griffin identifies such imagery as "backstage" photographs, non-combat images of troops and weaponry that dominated war coverage. While Griffin developed these concepts for wartime media relations, the visual vocabulary translates readily to peacetime contexts. The National Photo Company's military photographs operate in this same register. They present the military as an institution of procedural order, not an instrument of violence, showing formations, ceremonies, and inspections in place of combat or its consequences.

The cluster also reveals the logistical infrastructure of military presence in Washington. Tents appear frequently in the captions and document the temporary encampments that accompanied mobilisation. Images of canvas shelters and assembled equipment recorded the material conditions of camp life while also aestheticising them. A symmetrical photograph of soldiers standing before their tents turned the mundane routines of the encampment into a visual demonstration of military preparedness. American flags recur throughout the cluster and mark these spaces as sites of national significance, tying the activities they depict to the patriotic narratives that newspaper editors and their readers would immediately recognise.

What the cluster excludes proves as revealing as what it contains. Although the collection spans the First World War and its immediate aftermath, no images document wounded soldiers, battlefield destruction, or the human costs of military operations. This absence reflects the commercial constraints under which Herbert French's photographers operated. Newspapers and magazines sought images that could be printed without disturbing readers, and that would inspire confidence in military competence rather than raise questions about the costs of war. The resulting archive established a template for media-military visual relations that prioritised the performance of order, shaping how Americans pictured their military institutions for decades to come. 
<!-- #endregion -->

```python jdh={"module": "object", "object": {"source": ["Representative images from Topic 1 (Military Power)"], "type": "image"}} tags=["figure-topic1-images-*"]
# Figure 4: Topic 1 representative images
display(Image(filename=FIGURES_DIR / "topic_images" / "1.jpg", width=700))
print(f"Topic 1 — {topic_counts.get(1, 'N/A')} images")
print(f"Keywords: {get_topic_keywords(1, topics_data)}")
```

_In [Figure 4](#figure-topic1-images-*), the photographs depict military personnel in formal formations and official functions._

<!-- #region citation-manager={"citations": {"td1mf": [{"id": "17970137/MHHLIP9P", "source": "zotero"}]}} -->
**Official Governance (Topic 2)**

Topic 2 comprises 470 photographs centred on formal meetings and official proceedings. The cluster's defining keywords include "meeting," "conference," "discussion," "documents," "signing," and "chandeliers," capturing scenes of collective deliberation rather than individual authority. Where Topic 0 isolated bureaucrats at their desks, this cluster frames governance as a social act performed around conference tables. The captions consistently describe men "engaged in serious conversation" with documents "scattered" before them, suggesting that the camera's role was to witness the moment of decision rather than the person who made it. This shift from individual to group portraiture reflects the practical demands of the news industry, which required images showing that something had happened rather than simply who held office. A photograph of a committee in session could accompany a headline about policy debates in ways that a solitary desk portrait could not. 

Many of these photographs document international events, including the signing of the British War Loan in 1917, the French War Loan of the same year, and sessions with the Czechoslovakia and American Debt Commission in 1925. The cluster also contains images from congressional investigations, with the Teapot Dome hearings, steel strike testimonies, and immigration committee sessions all represented among the 470 photographs. The captions describe "men dressed in formal attire" arranged around tables with "large maps hanging on the wall," capturing Washington's position as a hub of postwar diplomacy and domestic oversight. These images served multiple audiences simultaneously. Domestic newspapers sought evidence of American engagement on the world stage while foreign press services required visual proof that agreements had been formalised. Congressional committees similarly needed photographic documentation that could be distributed to regional papers demonstrating that legislators were actively investigating matters of public concern. The National Photo Company's coverage of these events transformed abstract processes of negotiation and inquiry into concrete visual records that could circulate through the nation's print media networks.

The recurring presence of documents and signing implements in these photographs suggests an interest in the physical ritual of agreement, and this emphasis on the material trace of agreement reflects what Vilém Flusser identified as photography's capacity to make invisible processes visible <cite id="td1mf"><a href="#zotero%7C17970137%2FMHHLIP9P">(Flusser, 2000)</a></cite>. The signed document became proof that governance had occurred. What distinguishes these images from the desk portraits of Topic 0 is their focus on transaction rather than position. This temporal dimension gave the images particular value for newspapers covering breaking developments, as they could demonstrate not just who was involved but that a binding decision had been reached.
<!-- #endregion -->

```python jdh={"module": "object", "object": {"source": ["Representative images from Topic 2 (Official Governance)"], "type": "image"}} tags=["figure-topic2-images-*"]
# Figure 5: Topic 2 representative images
display(Image(filename=FIGURES_DIR / "topic_images" / "2.jpg", width=700))
print(f"Topic 2 — {topic_counts.get(2, 'N/A')} images")
print(f"Keywords: {get_topic_keywords(2, topics_data)}")
```

_In [Figure 5](#figure-topic2-images-*): Formal meetings, treaty signings, and committee proceedings characterise this cluster._

<!-- #region citation-manager={"citations": {"ol3pr": [{"id": "17970137/GIS7XRQC", "source": "zotero"}], "ub0th": [{"id": "17970137/KHP4KCI3", "source": "zotero"}]}} -->
**Washington's Built Environment (Topic 5)**

Beyond individual figures, the collection systematically recorded Washington's residential landscape. Topic 5, defined by keywords like "yard," "shutters," "driveway," and "manicured," captures domestic exteriors that reflect the city's housing inventory. As Claire Zimmerman notes, architectural photography in this period increasingly served commercial marketing functions alongside documentation <cite id="ub0th"><a href="#zotero%7C17970137%2FKHP4KCI3">(Zimmerman, 2014)</a></cite>. Similar to the most populated topics, the standardised framing of these houses (centred, symmetrical, and clear of clutter) reflects a desire to present the built environment as a site of order and respectability.

The captions generated for Topic 5 reveal a remarkably consistent visual vocabulary. House after house appears as a "two-story brick" structure with a "large front porch," "sloping roof," and "chimney." Properties are "surrounded by trees and shrubs" with "driveways leading up" to their entrances. This uniformity suggests that the National Photo Company developed standardised conventions for residential documentation, positioning cameras at consistent distances and angles to maximise the impression of substantial, well-maintained properties. The resulting images functioned less as architectural records than as visual testimonials to respectable living.

By extending the legitimating visual grammar to private neighbourhoods, the archive connects D.C.'s human subjects to the physical spaces they occupied. The orderly domesticity captured in Topic 5 represents the residential manifestation of the Progressive Era's broader urban vision. As Margaret Farrar argues, the architecture and design of a capital city serve to "create citizens," legitimising some groups while rendering others "out of place" <cite id="ol3pr"><a href="#zotero%7C17970137%2FGIS7XRQC">(Farrar, 2008)</a></cite>. The homes photographed in this cluster embody the aspirational ideal of citizen-homeownership that Progressive reformers championed as the foundation of democratic stability. A citizen who owned a well-maintained property with a front lawn and a porch had a literal stake in the civic order that the government buildings in Topic 0 represented.

What the cluster excludes proves as revealing as what it contains. The residential photographs overwhelmingly depict single-family homes with private yards, while Washington's alley dwellings and tenement blocks, which housed thousands of the city's working-class and African American residents, appear nowhere in the collection's domestic imagery. This selective documentation aligned with the priorities of Progressive Era housing reformers who characterised alley dwellings as breeding grounds for disease and moral degradation. Commercial clients seeking property documentation wanted images of marketable respectability, not documentary evidence of substandard housing that reformers were simultaneously campaigning to demolish.

The consistency of Topic 5's visual conventions also reflects the practical circumstances of commercial photography. Herbert French's photographers likely received commissions from real estate developers, mortgage companies, and individual homeowners who specified the kind of presentation they expected. Such clients would have rejected images that showed peeling paint, unmaintained yards, or neighbouring properties in poor condition. Over time, these commercial expectations produced a repertoire of compositional solutions that photographers applied routinely, such as centring the house in frame, including enough of the yard to suggest spaciousness, and capturing the property in favourable light that emphasised architectural details. The archive's residential photographs thus document not simply what Washington's housing looked like but what paying clients wanted it to look like.
<!-- #endregion -->

```python jdh={"module": "object", "object": {"source": ["Representative images from Topic 5 (Washington's Built Environment)"], "type": "image"}} tags=["figure-topic5-images-*"]
# Figure 6: Topic 5 representative images
display(Image(filename=FIGURES_DIR / "topic_images" / "5.jpg", width=700))
print(f"Topic 5 — {topic_counts.get(5, 'N/A')} images")
print(f"Keywords: {get_topic_keywords(5, topics_data)}")
```

_[Figure 6](#figure-topic5-images-*) shows standardised documentation of residential architecture across the capital._

<!-- #region citation-manager={"citations": {"06zma": [{"id": "17970137/TZ7Q5GA3", "source": "zotero"}]}} -->
### Who Was Seen, Who Was Silenced

The thematic clusters discussed above share a common feature: they depict spaces where women were systematically excluded in early twentieth-century society. Across all generated captions, male-gendered terms (man, men, he, his) outnumber female-gendered terms (woman, women, she, her) by nearly 4 to 1 (3.69:1). Within the main institutional topics, this imbalance becomes even more noticeable.

These numbers highlight the collection's gendered focus, revealing systematic biases that go beyond individual editorial choices. The disparity emerged from the convergence of multiple factors operating simultaneously. Herbert French's photographers followed their clients' assignments, clients requested subjects that would interest their audiences, and audiences had been conditioned to associate public authority with male figures. While individuals in this chain may not have necessarily intended to exclude women, the cumulative effect was a visual archive in which female presence became exceptional rather than ordinary. 

This pattern exemplifies what George Gerbner termed "symbolic annihilation," the systematic underrepresentation of marginalised groups that renders them invisible within the public visual record <cite id="06zma"><a href="#zotero%7C17970137%2FTZ7Q5GA3">(Gerbner, 1972)</a></cite>. The timing matters here. The collection spans the final push for women's suffrage, the ratification of the Nineteenth Amendment in 1920, and the decade following. One might expect photographs to reflect women's expanded civic presence after 1920, yet the algorithmic analysis reveals remarkable consistency in male dominance across the collection's entire timeframe. 

The way gender hierarchies are visually constructed also connects to race, although the algorithmic approach reveals these dynamics differently. The archive's documentation of events like the 1925 and 1926 Ku Klux Klan parades in Washington shows how commercial photography could make hate groups seem normal within public life. Furthermore, the collection's strong focus on institutional spaces, such as government buildings, military sites, and official gatherings, records locations that were either officially or unofficially segregated during Washington's Jim Crow era.

The archive largely omits documentation of African American community life in early twentieth-century Washington. The few images that do explicitly identify African American subjects consistently depict Black Washingtonians in service and labour roles, such as a nursemaid in a park and men shovelling snow and paving roads. Of the twelve photographs connected to Howard University, several document President Calvin Coolidge speaking at a 1924 graduation ceremony, capturing a white visitor rather than the institution's community. The remainder show campus buildings, a home economics class, and the Law Department, but these too are framed through the institution's utility to outside observers rather than its own intellectual life. Entirely absent are the vibrant institutions that defined Black Washington during this period, such as the businesses along U Street or the civic organisations that were shaping African American political mobilisation.

This gap in representation was not incidental. The National Photo Company's clients, primarily newspapers, magazines, and government agencies, operated within a segregated media system that determined which subjects warranted photographic documentation. The commercial priorities of Progressive Era photojournalism thus produced an archive that rendered African American civic life virtually invisible even as it recorded the public presence of those who opposed Black political participation. 
<!-- #endregion -->

```python
# Gender analysis: count gendered terms in captions

male_terms = r"\b(man|men|he|his|boy|boys|male|gentleman|gentlemen)\b"
female_terms = r"\b(woman|women|she|her|hers|girl|girls|female|lady|ladies)\b"

def count_matches(text, pattern):
    return len(re.findall(pattern, str(text).lower()))

captions_df["male_count"] = captions_df["caption"].apply(lambda x: count_matches(x, male_terms))
captions_df["female_count"] = captions_df["caption"].apply(lambda x: count_matches(x, female_terms))

total_male = captions_df["male_count"].sum()
total_female = captions_df["female_count"].sum()
ratio = total_male / total_female if total_female > 0 else float('inf')

# Presence in captions
has_male = (captions_df["male_count"] > 0).sum()
has_female = (captions_df["female_count"] > 0).sum()
male_only = ((captions_df["male_count"] > 0) & (captions_df["female_count"] == 0)).sum()

print("=== Gender Term Analysis ===")
print(f"Male term mentions: {total_male:,}")
print(f"Female term mentions: {total_female:,}")
print(f"Ratio: {ratio:.2f}:1")
print(f"\nCaptions mentioning men: {has_male:,} ({100*has_male/len(captions_df):.1f}%)")
print(f"Captions mentioning women: {has_female:,} ({100*has_female/len(captions_df):.1f}%)")
print(f"Captions with men only: {male_only:,} ({100*male_only/len(captions_df):.1f}%)")
```

### What Resists Classification

The structures of institutional power and gender disparity discussed earlier shape the photographs that form distinct topical groups; however, 23.7 per cent of the images (8,396 photographs) were placed in the noise cluster. Unlike traditional cataloguing that forces every item into fixed categories, HDBSCAN detects photographs that do not fit dominant trends, making evident the diversity and irregularity in large visual collections.

```python jdh={"module": "object", "object": {"source": ["Distribution of images between meaningful topics and the noise cluster"], "type": "image"}} tags=["figure-topic-distribution-noise-*"]
# Figure 7: Distribution of noise vs classified images
noise_count = topic_counts.get(-1, 0)
classified_count = len(results_df) - noise_count

plt.figure(figsize=(6, 6))
plt.pie([classified_count, noise_count], 
        labels=['Classified', 'Noise Cluster'], 
        autopct='%1.1f%%',
        colors=['steelblue', 'lightgray'],
        explode=[0, 0.05])
plt.title('Distribution of Images: Classified vs. Noise')
plt.tight_layout()
plt.show()

print(f"Classified images: {classified_count:,} ({100*classified_count/len(results_df):.1f}%)")
print(f"Noise cluster: {noise_count:,} ({100*noise_count/len(results_df):.1f}%)")
```

_In [Figure 7](#figure-topic-distribution-noise-*), nearly a quarter of all photographs resist topical categorisation._

```python jdh={"module": "object", "object": {"source": ["Representative images from the Noise Cluster (Topic -1)"], "type": "image"}} tags=["figure-images-noise-*"]
# Figure 8: Noise cluster representative images
display(Image(filename=FIGURES_DIR / "topic_images" / "-1.jpg", width=700))
print(f"Noise Cluster (Topic -1) — {noise_count} images")
print(f"Keywords: {get_topic_keywords(-1, topics_data)}")
```

_In [Figure 8](#figure-images-noise-*), the photographs resist topical categorisation, demonstrating the diversity of content that falls outside the collection's dominant visual logics._

Notably, the noise cluster may include precisely those photographs that oppose the primary visual themes of the collection, images whose meanings are heavily dependent on context or too idiosyncratic to be captured through automated clustering. Several examples demonstrate this diversity. A photograph captioned "Mrs. Geo. Oakley Totten" shows "a woman sitting at a table with a sculpture of a ballerina in front of her... holding a paintbrush and appears to be in the process of painting the sculpture." This is an artistic domestic scene that goes beyond the collection's usual themes of institutional masculinity. Another image, simply captioned "Children," displays "a group of children and a man standing in front of a building... of different ages and ethnicities," a subject that challenges the conventions of formal portraiture common in most topics.

Therefore, researchers should view the noise cluster not just as leftover data but as an important category that defines the limits of automated subject organisation.

<!-- #region tags=["hermeneutics"] -->
# What the Method Cannot See

It is important to consider the findings of this study alongside four broad limitations that affect both the historical claims and the methodological contributions: caption accuracy, the inherent limitations of visual-textual translation, layered biases in the source material, and collection scope.

The first limitation involves caption accuracy. As noted before, Florence-2 performs well with modern photographs but may struggle with historical images. Unfamiliar objects, period-specific clothing, and damaged negatives can lead to inaccurate descriptions. Sometimes, misidentifications may also happen, such as a photograph labelled "First National Bank of So. Md. Marlborough" (LCCN2016826472), which produces a caption stating that the sign reads "Bank of America", even though the building's signage plainly reads First National Bank of Southern Maryland (Figure 9). A study of sample photographs confirmed that Florence-2 generally produces reasonable descriptions of overall scene composition, though a comprehensive evaluation of caption accuracy against ground truth annotations across the entire corpus was not conducted. Consequently, some topic clusters may reflect model errors rather than actual visual features.
<!-- #endregion -->

```python jdh={"module": "object", "object": {"source": ["First National Bank of So. Md. Marlborough (LCCN2016826472)"], "type": "image"}} tags=["figure-bank-marlborough-*"]
# Figure 9: Caption accuracy limitation example
display(Image(filename=FIGURES_DIR / "first_national_bank_caption_error.jpg", width=500))
print("Figure 9: First National Bank of So. Md. Marlborough (LCCN2016826472).")
print("Florence-2 incorrectly identifies the signage as 'Bank of America.'")
```

_Florence-2 incorrectly identifies the signage in [Figure 9](#figure-bank-marlborough-*) as "Bank of America," demonstrating how vision-language models struggle with historical images containing period-specific text and unfamiliar institutional names._

<!-- #region citation-manager={"citations": {"jor7p": [{"id": "17970137/KIB89ZN5", "source": "zotero"}]}} -->
The second limitation relates to visual-textual translation itself. While the method detailed here can better capture semantic information than techniques that rely only on low-level feature extraction, no automated method can capture aspects of meaning that defy verbal description. Photographs convey emotional responses, artistic arrangements of elements, and the physical sensation of viewing an image that go beyond what natural language captions can express. For instance, a photograph might evoke awe or unease through its lighting and shadows, or create a feeling of intimacy through its framing, but these affective dimensions are difficult to translate. Thus, this method may still miss the broader interpretive layers that historians might draw from contextual knowledge or visual literacy. 

The third limitation involves layered biases. Multiple levels of bias influence both the source material and the algorithmic analysis. As Amanda Wasielewski warns, "the great fallacy of quantitative research is usually not the method itself, but rather the belief in the purity of the data and its lack of bias. Counting a biased sample reproduces that bias" <cite id="jor7p"><a href="#zotero%7C17970137%2FKIB89ZN5">(Wasielewski, 2023)</a></cite>. The original National Photo Company collection reflects the commercial priorities and social perspectives of Herbert E. French and his photographers, who focused on subjects that would sell newspapers and appeal to their audience. This creates an initial bias towards certain types of images. Additionally, the Library of Congress's digitisation process may have prioritised certain materials, meaning the digitised subset may not fully reflect the agency's original commercial priorities.

A fourth limitation concerns collection scope. The noise cluster analysis revealed an unexpected finding: a photograph of concentration camp uniforms (LCCN90709928) that clearly postdates the National Photo Company's operations (1909-1932). While the collection contains numerous photographs relating to Germany from the World War I era, this concentration camp image dates from the Nazi period, well after the company ceased operations. The image was likely part of Herbert French's personal collection, which the Library of Congress acquired alongside his commercial archive. Its isolation as the sole post-1932 German photograph suggests an anomalous inclusion rather than systematic collecting beyond the company's active years (Figure 10). This discovery suggests that the "National Photo Company Collection" as presented through the Library of Congress interface may not represent a unified corpus but rather an aggregation of materials with complex and sometimes opaque provenance histories. While such anomalies do not invalidate the overall analysis (the concentration camp photograph was correctly identified as noise and excluded from topic modelling), researchers working with institutional collections should remain attentive to the possibility that named collections may contain materials beyond their stated scope.
<!-- #endregion -->

```python jdh={"module": "object", "object": {"source": ["Concentration camp uniforms (LCCN90709928)"], "type": "image"}} tags=["figure-camp-uniforms-*"]
# Figure 10: Archival provenance limitation example
display(Image(filename=FIGURES_DIR / "concentration_camp_provenance.jpg", width=500))
print("Figure 10: Concentration camp uniforms (LCCN90709928).")
print("This photograph postdates the National Photo Company's operations.")
```

_This photograph ([Figure 10](#figure-camp-uniforms-*)) which clearly postdates the National Photo Company's operations (1909-1932), was found within the collection's noise cluster. Its presence demonstrates how computational methods can reveal archival provenance complications invisible through traditional interfaces._

<!-- #region citation-manager={"citations": {"4799w": [{"id": "17970137/UB9J98RS", "source": "zotero"}], "7zf1r": [{"id": "17970137/TZ7Q5GA3", "source": "zotero"}], "mfey9": [{"id": "17970137/YFVNDSCE", "source": "zotero"}], "rcmcb": [{"id": "17970137/TZ7Q5GA3", "source": "zotero"}]}} tags=["hermeneutics"] -->
## Conclusion

Overall, this study shows how algorithmic methods can uncover visual representation trends that are hard to identify through traditional archival research alone. Analysing 35,368 photographs revealed 527 distinct subject clusters, a nearly four-to-one male-to-female term ratio, and a 23.7 per cent noise cluster. This quantitative data offers new insights into how commercial photography influenced visual authority during the Progressive Era in Washington. These findings suggest that the National Photo Company acted as a "visual infrastructure of state legitimation," making governmental authority appear natural, orderly, and inevitable through repeated use of specific compositional conventions.

The clear gender disparities highlight patterns that scholars in visual culture have long studied, yet rarely measured on a large scale. The nearly four-to-one overall ratio of male to female references (3.69:1), which rises to over one thousand to one in the office and document-signing topics, demonstrates the concept of "symbolic annihilation," which George Gerbner introduced and Gaye Tuchman extended to the representation of women <cite id="rcmcb"><a href="#zotero%7C17970137%2FTZ7Q5GA3">(Gerbner, 1972)</a></cite>, <cite id="4799w"><a href="#zotero%7C17970137%2FUB9J98RS">(Tuchman, 1978)</a></cite>. The dearth of African American subjects in the collection reflects the same process: the systematic underrepresentation that renders marginalised groups absent from media representation <cite id="7zf1r"><a href="#zotero%7C17970137%2FTZ7Q5GA3">(Gerbner, 1972)</a></cite>.

Methodologically, this study presents a reproducible pipeline that links visual archives with established text analysis techniques. Combining vision-language captioning with neural topic modelling enables researchers to examine visual collections using familiar digital humanities methods. Furthermore, the deterministic settings ensure that results can be verified and lay a foundation for scholarly discussion about both the methods and their outcomes.

The explicit documentation of the noise cluster (the 23.7 per cent of photographs that resisted topical categorisation) frames algorithmic classification as a complement, not a replacement, for traditional historical methods. By recognising what defies categorisation instead of forcing every image into predefined topics, this approach creates starting points for humanistic inquiry rather than claiming definitive classifications. As Taylor Arnold and Lauren Tilton argue, computational methods work "not at the expense of close viewing. In combination with subject matter expertise and subsequent close analyses... [they allow] scholars to ask and address new and existing questions" <cite id="mfey9"><a href="#zotero%7C17970137%2FYFVNDSCE">(Arnold &#38; Tilton, 2019)</a></cite>. The topics identified here serve as tentative maps, encouraging historians to examine representative images, evaluate cluster coherence, and use algorithmic groupings as entry points for deeper archival exploration.

## Future Directions

The strengths and limitations discussed above point to several promising directions for future research. First, the methodology developed here could be applied to other collections. Comparing these collections would determine whether the patterns identified, such as the focus on institutional authority and the marginalisation of women and minorities, are unique to Herbert E. French's work or reflect broader trends in early twentieth-century American commercial photography.

Second, for historians looking to use these findings, the topics can serve as curated subsets suitable for focused research. A historian studying women's representation in Progressive Era Washington might examine the 20 per cent of captions that mention women to identify which topics include female subjects and the contexts in which they appear. A researcher interested in military visual culture could analyse Topic 1's 598 photographs as a defined set for detailed study of uniform styles, rank insignia, and compositional patterns.

Finally, the noise cluster itself also deserves further study. The 8,396 photographs that resisted topical classification may feature subjects and visual styles that fall outside mainstream commercial standards and could complicate or challenge the main patterns identified in this research.
<!-- #endregion -->

<!-- #region tags=["hidden"] -->
## References

<!-- BIBLIOGRAPHY START -->
<div class="csl-bib-body">
  <div class="csl-entry"><i id="zotero|17970137/YFVNDSCE"></i>Arnold, T., &#38; Tilton, L. (2019). Distant Viewing: Analyzing Large Visual Corpora. <i>Digital Scholarship in the Humanities</i>, <i>34</i>(Supplement_1), i3–i16. <a href="https://doi.org/10.1093/llc/fqz013">https://doi.org/10.1093/llc/fqz013</a></div>
  <div class="csl-entry"><i id="zotero|17970137/24TN4JIG"></i>Arnold, T., &#38; Tilton, L. (2022). <i>ADDI: Methods Paper</i>. <a href="https://distant-viewing.github.io/addi/">https://distant-viewing.github.io/addi/</a></div>
  <div class="csl-entry"><i id="zotero|17970137/H5IBYMNW"></i>Arnold, T., &#38; Tilton, L. (2024). <i>Explainable Search and Discovery of Visual Cultural Heritage Collections with Multimodal Large Language Models</i> (arXiv:2411.04663). arXiv. <a href="https://doi.org/10.48550/arXiv.2411.04663">https://doi.org/10.48550/arXiv.2411.04663</a></div>
  <div class="csl-entry"><i id="zotero|17970137/FU3ZC6ZL"></i>Blei, D. M., Ng, A. Y., &#38; Jordan, M. I. (2003). Latent Dirichlet Allocation. <i>Journal of Machine Learning Research</i>, <i>3</i>, 993–1022.</div>
  <div class="csl-entry"><i id="zotero|17970137/N5YY7JRY"></i>Bourdieu, P. (1990). <i>Photography: A Middle-Brow Art</i>. Stanford University Press.</div>
  <div class="csl-entry"><i id="zotero|17970137/ZXJ4FHK6"></i>Cohen, M. (1999). <i>The Sentimental Education of the Novel</i>. Princeton University Press. <a href="https://doi.org/10.1515/9780691188249">https://doi.org/10.1515/9780691188249</a></div>
  <div class="csl-entry"><i id="zotero|17970137/3I47GI3L"></i>Crawford, K., &#38; Paglen, T. (2021). Excavating AI: The Politics of Images in Machine Learning Training Sets. <i>AI &#38; SOCIETY</i>, <i>36</i>(4), 1105–1116. <a href="https://doi.org/10.1007/s00146-021-01162-8">https://doi.org/10.1007/s00146-021-01162-8</a></div>
  <div class="csl-entry"><i id="zotero|17970137/VZ7HGW5B"></i>Devlin, J., Chang, M.-W., Lee, K., &#38; Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In J. Burstein, C. Doran, &#38; T. Solorio (Eds.), <i>Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)</i> (pp. 4171–4186). Association for Computational Linguistics. <a href="https://doi.org/10.18653/v1/N19-1423">https://doi.org/10.18653/v1/N19-1423</a></div>
  <div class="csl-entry"><i id="zotero|17970137/GIS7XRQC"></i>Farrar, M. E. (2008). <i>Building the Body Politic: Power and Urban Space in Washington, D.C.</i> University of Illinois Press.</div>
  <div class="csl-entry"><i id="zotero|17970137/MHHLIP9P"></i>Flusser, V. (2000). <i>Towards a Philosophy of Photography</i>. Reaktion.</div>
  <div class="csl-entry"><i id="zotero|17970137/TZ7Q5GA3"></i>Gerbner, G. (1972). Violence in Television Drama: Trends and Symbolic Functions. In G. A. Comstock &#38; Eli. A. Rubinstein (Eds.), <i>Television and Social Behavior: Media Content and Control</i> (Vol. 1, pp. 28–187). U.S. Government Printing Office. <a href="https://www.ojp.gov/pdffiles1/Digitization/148976NCJRS.pdf">https://www.ojp.gov/pdffiles1/Digitization/148976NCJRS.pdf</a></div>
  <div class="csl-entry"><i id="zotero|17970137/BVRKVU5J"></i>Green, C. M. (1963). <i>Washington, Capital City, 1879–1950</i> (Vol. 2). Princeton University Press.</div>
  <div class="csl-entry"><i id="zotero|17970137/7VSCBJ52"></i>Griffin, M. (2010). Media Images of War. <i>Media, War &#38; Conflict</i>, <i>3</i>(1), 7–41. <a href="https://doi.org/10.1177/1750635210356813">https://doi.org/10.1177/1750635210356813</a></div>
  <div class="csl-entry"><i id="zotero|17970137/GX7MI9NR"></i>Grootendorst, M. (2020). <i>KeyBERT: Minimal Keyword Extraction with BERT</i>. <a href="https://github.com/MaartenGr/KeyBERT">https://github.com/MaartenGr/KeyBERT</a></div>
  <div class="csl-entry"><i id="zotero|17970137/YI6YHSEP"></i>Grootendorst, M. (2022). <i>BERTopic: Neural Topic Modeling with a Class-Based TF-IDF Procedure</i> (arXiv:2203.05794). arXiv. <a href="https://doi.org/10.48550/arXiv.2203.05794">https://doi.org/10.48550/arXiv.2203.05794</a></div>
  <div class="csl-entry"><i id="zotero|17970137/LAQHGXKA"></i>Hall, S. (1997). <i>Representation: Cultural Representations and Signifying Practices</i>. SAGE Publications.</div>
  <div class="csl-entry"><i id="zotero|17970137/KL9FD2CB"></i>ImageMagick Studio LLC. (2024). <i>ImageMagick</i> (7.1.1). <a href="https://imagemagick.org">https://imagemagick.org</a></div>
  <div class="csl-entry"><i id="zotero|17970137/V6LU8JJE"></i>Impett, L., &#38; Offert, F. (2024). There Is a Digital Art History. <i>Visual Resources</i>, <i>38</i>(2), 186–209. <a href="https://doi.org/10.1080/01973762.2024.2362466">https://doi.org/10.1080/01973762.2024.2362466</a></div>
  <div class="csl-entry"><i id="zotero|17970137/799URSTD"></i>Library of Congress. (n.d.). <i>About this Collection | National Photo Company Collection | Digital Collections | Library of Congress</i> [Collection]. Library of Congress. Retrieved May 28, 2025, from <a href="https://www.loc.gov/collections/national-photo-company/about-this-collection/">https://www.loc.gov/collections/national-photo-company/about-this-collection/</a></div>
  <div class="csl-entry"><i id="zotero|17970137/2DPNW9Q8"></i>Library of Congress Research Guides. (n.d.). <i>National Photo Company Collection</i>. <a href="https://guides.loc.gov/national-photo-company">https://guides.loc.gov/national-photo-company</a></div>
  <div class="csl-entry"><i id="zotero|17970137/KHSFXVNA"></i>Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., &#38; Zitnick, C. L. (2014). Microsoft COCO: Common Objects in Context. In D. Fleet, T. Pajdla, B. Schiele, &#38; T. Tuytelaars (Eds.), <i>Computer Vision – ECCV 2014</i> (pp. 740–755). Springer International Publishing. <a href="https://doi.org/10.1007/978-3-319-10602-1_48">https://doi.org/10.1007/978-3-319-10602-1_48</a></div>
  <div class="csl-entry"><i id="zotero|17970137/ZKGMP49X"></i>Manovich, L. (2015). Data Science and Digital Art History. <i>International Journal for Digital Art History</i>, <i>1</i>. <a href="https://doi.org/10.11588/dah.2015.1.21631">https://doi.org/10.11588/dah.2015.1.21631</a></div>
  <div class="csl-entry"><i id="zotero|17970137/IYV2WM3S"></i>Meeks, E., &#38; Weingart, S. B. (2012). The Digital Humanities Contribution to Topic Modeling. <i>Journal of Digital Humanities</i>, <i>2</i>(1), 1–6. <a href="https://journalofdigitalhumanities.org/2-1/dh-contribution-to-topic-modeling/">https://journalofdigitalhumanities.org/2-1/dh-contribution-to-topic-modeling/</a></div>
  <div class="csl-entry"><i id="zotero|17970137/LS4UG6JJ"></i>Mitchell, W. J. T. (2005). There Are No Visual Media. <i>Journal of Visual Culture</i>, <i>4</i>(2), 257–266. <a href="https://doi.org/10.1177/1470412905054673">https://doi.org/10.1177/1470412905054673</a></div>
  <div class="csl-entry"><i id="zotero|17970137/9CT6669J"></i>Moretti, F. (2000). Conjectures on World Literature. <i>New Left Review</i>, <i>2</i>(1), 54–68. <a href="https://doi.org/10.64590/hxj">https://doi.org/10.64590/hxj</a></div>
  <div class="csl-entry"><i id="zotero|17970137/6BIEITXL"></i>Natanson, B. O. (2007). Worth a Billion Words? Library of Congress Pictures Online. <i>Journal of American History</i>, <i>94</i>(1), 99–111. <a href="https://doi.org/10.2307/25094779">https://doi.org/10.2307/25094779</a></div>
  <div class="csl-entry"><i id="zotero|17970137/4E7L8EWK"></i>Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., &#38; Sutskever, I. (2021). <i>Learning Transferable Visual Models From Natural Language Supervision</i> (arXiv:2103.00020). arXiv. <a href="https://doi.org/10.48550/arXiv.2103.00020">https://doi.org/10.48550/arXiv.2103.00020</a></div>
  <div class="csl-entry"><i id="zotero|17970137/X3Q5JQUX"></i>Reps, J. W. (1967). <i>Monumental Washington: The Planning and Development of the Capital Center</i>. Princeton University Press.</div>
  <div class="csl-entry"><i id="zotero|17970137/7G4M5KXE"></i>Robertson, S., &#38; Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. <i>Foundations and Trends in Information Retrieval</i>, <i>3</i>(4), 333–389. <a href="https://doi.org/10.1561/1500000019">https://doi.org/10.1561/1500000019</a></div>
  <div class="csl-entry"><i id="zotero|17970137/CJ7LNUTM"></i>Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., Berg, A. C., &#38; Fei-Fei, L. (2015). ImageNet Large Scale Visual Recognition Challenge. <i>International Journal of Computer Vision</i>, <i>115</i>(3), 211–252. <a href="https://doi.org/10.1007/s11263-015-0816-y">https://doi.org/10.1007/s11263-015-0816-y</a></div>
  <div class="csl-entry"><i id="zotero|17970137/HBWAKTCH"></i>Sekula, A. (1986). The Body and the Archive. <i>October</i>, <i>39</i>, 3–64. <a href="https://doi.org/10.2307/778312">https://doi.org/10.2307/778312</a></div>
  <div class="csl-entry"><i id="zotero|17970137/TK7HWQH7"></i>Smeulders, A. W. M., Worring, M., Santini, S., Gupta, A., &#38; Jain, R. (2000). Content-Based Image Retrieval at the End of the Early Years. <i>IEEE Transactions on Pattern Analysis and Machine Intelligence</i>, <i>22</i>(12), 1349–1380. <a href="https://doi.org/10.1109/34.895972">https://doi.org/10.1109/34.895972</a></div>
  <div class="csl-entry"><i id="zotero|17970137/GEYL3A6G"></i>Solaroli, M. (2016). The Rules of a Middle-Brow Art: Digital Production and Cultural Consecration in the Global Field of Professional Photojournalism. <i>Poetics</i>, <i>59</i>, 50–66. <a href="https://doi.org/10.1016/j.poetic.2016.09.001">https://doi.org/10.1016/j.poetic.2016.09.001</a></div>
  <div class="csl-entry"><i id="zotero|17970137/CCPYPJNV"></i>Sontag, S. (1990). <i>On Photography</i>. Picabor.</div>
  <div class="csl-entry"><i id="zotero|17970137/DSFJSTTI"></i>Tagg, J. (1988). <i>The Burden of Representation: Essays on Photographies and Histories</i>. Macmillan. <a href="https://doi.org/10.1007/978-1-349-19355-4">https://doi.org/10.1007/978-1-349-19355-4</a></div>
  <div class="csl-entry"><i id="zotero|17970137/F4X7LWSN"></i>Thomas, J., &#38; Testini, I. (2024). Capturing Captions: Using AI to Identify and Analyse Image Captions in a Large Dataset of Historical Book Illustrations. <i>Digital Humanities Quarterly</i>, <i>18</i>(3). <a href="https://doi.org/10.63744/7a993yjxzfac">https://doi.org/10.63744/7a993yjxzfac</a></div>
  <div class="csl-entry"><i id="zotero|17970137/UB9J98RS"></i>Tuchman, G. (1978). Introduction: The Symbolic Annihilation of Women by the Mass Media. In G. Tuchman, A. K. Daniels, &#38; J. Benét (Eds.), <i>Hearth and Home: Images of Women in the Mass Media</i> (pp. 3–38). Oxford University Press.</div>
  <div class="csl-entry"><i id="zotero|17970137/9SF4LTTE"></i>van Noord, N. (2022). A Survey of Computational Methods for Iconic Image Analysis. <i>Digital Scholarship in the Humanities</i>, <i>37</i>(4), 1316–1338. <a href="https://doi.org/10.1093/llc/fqac003">https://doi.org/10.1093/llc/fqac003</a></div>
  <div class="csl-entry"><i id="zotero|17970137/YXL9CHS2"></i>Wang, W., Bao, H., Huang, S., Dong, L., &#38; Wei, F. (2021). MiniLMv2: Multi-Head Self-Attention Relation Distillation for Compressing Pretrained Transformers. In C. Zong, F. Xia, W. Li, &#38; R. Navigli (Eds.), <i>Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021</i> (pp. 2140–2151). Association for Computational Linguistics. <a href="https://doi.org/10.18653/v1/2021.findings-acl.188">https://doi.org/10.18653/v1/2021.findings-acl.188</a></div>
  <div class="csl-entry"><i id="zotero|17970137/KIB89ZN5"></i>Wasielewski, A. (2023). <i>Computational Formalism: Art History and Machine Learning</i>. MIT Press. <a href="https://doi.org/10.7551/mitpress/14268.001.0001">https://doi.org/10.7551/mitpress/14268.001.0001</a></div>
  <div class="csl-entry"><i id="zotero|17970137/2SRZSPH2"></i>Wevers, M., &#38; Smits, T. (2020). The Visual Digital Turn: Using Neural Networks to Study Historical Images. <i>Digital Scholarship in the Humanities</i>, <i>35</i>(1), 194–207. <a href="https://doi.org/10.1093/llc/fqy085">https://doi.org/10.1093/llc/fqy085</a></div>
  <div class="csl-entry"><i id="zotero|17970137/Q3CB4DLN"></i>Wikimedia Commons. (n.d.). <i>Category: National Photo Company Collection</i>. <a href="https://commons.wikimedia.org/wiki/Category:National_Photo_Company_Collection">https://commons.wikimedia.org/wiki/Category:National_Photo_Company_Collection</a></div>
  <div class="csl-entry"><i id="zotero|17970137/FAEP32ZA"></i>Xiao, B., Wu, H., Xu, W., Dai, X., Hu, H., Lu, Y., Zeng, M., Liu, C., &#38; Yuan, L. (2023). <i>Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks</i> (arXiv:2311.06242). arXiv. <a href="https://doi.org/10.48550/arXiv.2311.06242">https://doi.org/10.48550/arXiv.2311.06242</a></div>
  <div class="csl-entry"><i id="zotero|17970137/KHP4KCI3"></i>Zimmerman, C. (2014). <i>Photographic Architecture in the Twentieth Century</i>. University of Minnesota Press.</div>
</div>
<!-- BIBLIOGRAPHY END -->
<!-- #endregion -->
