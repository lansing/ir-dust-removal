# ir-dust-removal
## Remove dust using infrared channel in Silverfast 64 bit DNG files


Outputs cleaned TIFF along with a TIFF of the mask used. Intended for use along with Negative Lab Pro.


Inspiration: nlp forum discussion:

https://forums.negativelabpro.com/t/requesting-a-proper-isrd-pipeline/3701/17?page=2


## Usage


```
pip install -r requirements.txt

python remove-dust.py [-h] [-c MAX_COVERAGE] [-b BORDER_BUFFER] [-m] input
```

## Workflow:

1. read IR channel using PyUnraw, write out as TIFF
2. read RGB channels
3. build mask: sweep through range of thresholds, stop when mask
   coverage at MAX_COVERAGE. write mask TIFF.
4. clean channels one at a time using OpenCV inpainting
5. write TIFF of cleaned image


TODO: write out a silverfast DNG (NLP/Lightroom seems to handle this marginally better)


