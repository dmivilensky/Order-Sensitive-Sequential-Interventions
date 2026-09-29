ASSISTments 2009-2010 skill-builder data (not redistributed: the ASSISTments terms of use forbid passing the data on).

Download the original (not the "corrected") skill-builder file from
https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010
and accept the terms of use there. Cite Feng, Heffernan & Koedinger (2009), User Modeling and
User-Adapted Interaction 19(3):243-266, doi:10.1007/s11257-009-9063-7, and the URL above.

The paper uses the first 192,000 data rows of the file (it is sorted by skill id; this covers the
skills with ids <= 70). Create the subset with

    python3 -c "import pandas as pd; pd.read_csv('skill_builder_data.csv', encoding='latin1', nrows=192000, low_memory=False).to_csv('assistments_subset.csv', index=False)"

and put assistments_subset.csv in this folder.
