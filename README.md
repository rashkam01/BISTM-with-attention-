# BISTM-with-attention-
BILSTM with attention mechnism for relation classification
In an automotive part manufacturing industry, there is possibility of product recalls due to a variety of reasons. 
Some of the reasons include internal or external part damage, mistakes in packaging process, poor internal management etc. 
However, identifying quality defects is often the most difficult and the most critical for the producer.

There are various problem solving process to handle product returns. 
A problem is defined as a deviation from a defined target situation, 
there are many ways to for problem solving and one such method of problem solving is 8D process, 
it defines 8 steps which includes not just solving the problem but also a root cause analysis 
so that the problem does not reoccur again and also a containment action to limit the problem and resume normal operations. 
However, each and every product already has an fmea, an fmea pre-defines defines risk analysis, 
or the probable risk and failures a product can have, and there is a possiblity the the root cause might be defined in FMEA.

Since, most of it is in textual format and need domain expertise to search through an FMEA or 
perform a root cause analysis of the problem from scratch. Therefore, structuring of fmea contents 
can significantly improve the quality of 8D process and can also act as the basis for other applications.

A common approach for structuring of documents is to build a Named-entity recognition model for 
detecting the parts and subparts in each and every sentence of an fmea document. 
An addition to named entity recognition is also relationship extraction where we identify the 
relation of potential failure or function of the entity.
