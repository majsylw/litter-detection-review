# Trash detection - review of useful resources
A list of useful resources in the litter detection (mainly plastics)
* [Datasets](https://github.com/majsylw/litter-detection-review/blob/main/README.md#Datasets)
* [Papers](https://github.com/majsylw/litter-detection-review/blob/main/README.md#Papers)

# Datasets

- **[TACO](http://tacodataset.org/)** is an open image dataset of waste in the wild. It contains photos of litter taken under diverse environments, from tropical beaches to London streets. These images are manually labeled and segmented according to a hierarchical taxonomy to train and evaluate object detection algorithms. The best way to know TACO is to explore our dataset:
  - statistics: 1500 images with 4784 annotations and 3167 new images that need to be annotated. These annotations are labeled in 60 categories which belong to 28 super (top) categories, more you can find [HERE](http://tacodataset.org/stats),
  - annotations: masks provided in COCO format for instance segmentation task,
  - characteristics: images mainly came from Spain and its background is divided by 6 scene categories (photos mainly taken with rubbish in the background of sand, dirt, rocks)
  - download: tlist of all URLs for unlabelled and labelled TACO images is available in [all_image_urls.csv](https://github.com/pedropro/TACO/blob/master/data/all_image_urls.csv),
  - licence: Both the images and annotations provided by this dataset are all under free copyright licences. While the annotations are licenced under CC BY 4.0, images may be under different specific public licences since these were obtained from different sources and users. The licence along with the original URL of each individual image are referenced in the accompanying annotation file. If the licence entry is missing, then this is by default: CC BY 4.0.
  - paper: [TACO: Trash Annotations in Context for Litter Detection](https://arxiv.org/pdf/2003.06975.pdf), March 2020,
  - year: 2019 -- 2020.
- **[open-litter-map](https://openlittermap.com/)**
  - statistics: around 100 000 photos of litter taken around the world, 100+ types of litter have been pre-defined by OpenLitterMap, which are mapped by several behavioral categories of related-waste, coastal litter, litter art and global corporations
  - annotations: annotated with the classes (mainly one, but also has multilabel cases) per image,
  - characteristics: photos were taken around the world in America, Europe, Africa,  North Asia, and Australia (photos have also some geospatial data),
  - download: ?
  - licence: ODbL – Open Database Licence, see [openlittermap.com/term](openlittermap.com/term),
  - paper: (https://opengeospatialdata.springeropen.com/articles/10.1186/s40965-018-0050-y), June 2018,
  - year: website and aplication was registered in 27th July 2015, launched 15th April 2017.
- **trashnet**
  - statistics:
  - annotations:
  - characteristics:
  - download:
  - paper:
  - year: 
- **[WaDaBa](http://wadaba.pcz.pl/)** is waste database creating by taking a photo of a platform, on which are put objects - wastes in two type of light sources: the fluorescent lamp and LED-bulb. Waste were acquisition and photographed by four months in order to collect as the most typical kinds of the municipal waste:
  - statistics: 4 000 of photographs of different type of plastics (PET, PE-HD, PVC, PE-LD, PP, PS and other) were created in the database,
  - annotations: label type (category) per image,
  - characteristics: authors prepared 10 photographs with differ in the angle of the turnover for every object (in the vertical axis). Next the object was damaged to varying degrees: small, medium and large. For each type of destruction have been made 10 photographs. So considering all variants for every object 40 photographs were taken, multiplying it by the number of 100 objects (one bject per image),
  - download: if you want get the dataset, you should sent special form to aouthors.
  - paper: [PET waste clasification method and Plastic Waste DataBase WaDaBa](http://wadaba.pcz.pl/JBJP_ipc2017.pdf), September 2017,
  - year: 2017.
- **[Glassense-Vision](http://www.slipguru.unige.it/Data/glassense_vision/)** is a set of data we acquired and annotated to the purpose of providing a quantitative and repeatable assessment of the proposed method. The dataset includes 7 different use cases, meaning different object categories, where for each one of them we provide training (reference images used also to build dictionaries) and test images:
  - statistics:
  - annotations: manually annotated with the class and the object instance,
  - characteristics: each image shows a different type of litter, acquired on a uniform background, all images have been stored at a resolution of 665x1182 pixels,
  - download: user can download zip files of seven object categories by cliking on the links in Download section at [page](http://www.slipguru.unige.it/Data/glassense_vision/).
  - licence: the dataset is provided "as is" and without any express or implied warranties, including, without limitation, the implied warranties of merchantability and fitness for a particular purpose.
  - paper: [Hands on recognition: adding a vision touch to tactile and sound perception for visually impaired users](https://dl.acm.org/doi/10.1145/3060056), August 2017,
  - year: 2017.

# Papers

## Classification

## Detection
