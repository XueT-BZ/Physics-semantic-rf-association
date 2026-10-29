## LoED: The LoRaWAN at the Edge Dataset

This repository contains dataset from nine LoRaWAN gateways collected in an urban environment. The dataset contains raw payload information, along with other metadata from the gateway. The dataset can be cited at:

> **Dataset: LoED: The LoRaWAN at the Edge Dataset**  
> Laksh Bhatia, Michael Breza, Ramona Marfievici, Julie A. McCann  
> Proceedings of the Third Workshop on Data Acquisition To Analysis (DATA '20), 2020

### Information about the gateways

A detailed information about all gateways:

| Gateway ID | Description of location | Latitude | Longitude | Altitude (meters) | Model | Number of days | Total messages | max messages in a day | avg messages per day | 
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
| 00000f0c210281c4 | Dense outdoor area, on top of a building | 51.506900 | -0.1160894 | 25 | Cisco Wireless Gateway for LoRaWAN | 19 | 1326687 | 82781 | 69826 |
| 00000f0c22433141 | Roof of a low building in a non-dense area | 51.49120 | -0.12774 | 20 | Cisco Wireless Gateway for LoRaWAN |  36 | 144777 | 6579 | 4022 |
| 00000f0c210721f2 | Top of a building in a very dense area and large open spaces | 51.50766 | -0.0989 | 40 |  Cisco Wireless Gateway for LoRaWAN |  56 | 5757575 | 121368 | 102814 |
| 00000f0c224331c4 | Indoor in the ground floor a building, surrounded by buildings | 51.5046 | -0.11119 | 2 | Cisco Wireless Gateway for LoRaWAN |  15 | 17029 | 1625 | 1135 |
| 00800000a0001914 | Deployed inside a university building | 51.49896 | -0.17801 | 5 | Multitech MTCDT-H5-246A-868-EU-GB |  573 | 76706 | 2366 | 134 |
| 00800000a0001793 | Deployed inside a university building | 51.49843 |  -0.17823 | 5 | Multitech MTCDT-H5-246A-868-EU-GB |  552 | 186592 | 9596 | 338 |
| 00800000a0001794 | Deployed inside a university building | 51.49896 | -0.17801  | 5 | Multitech MTCDT-H5-246A-868-EU-GB |  17 | 61080 | 4810 | 3593 |
| 7276ff002e062804 | Deployed on top of a tall university building, with large open spaces |  51.49904 | -0.1764 | 65 | Kerlink Wirnet Station V2 |  131 | 1201916 | 15254 | 9175 |
| 0000024b0b031c97 | Urban area, top of building, dense deployment | 51.52183 | -0.135 | 66 | Kerlink Wirnet Station V2 |  131 | 2490639 | 25708 | 19013 |

### Files in the folder
* **Laksh_Bhatia_LoED_LoRaWAN_at_the_edge.pdf**   
	A two page abstract describing the dataset
* **LoED_LoRaWAN_at_edge_dataset.zip**   
	One file with all data files. Data files are in **dd\_mm\_yyyy.csv** format with one file for every day of the collection campaign
* **LoED_LoRaWAN_at_edge_dataset-SAMPLE.zip**   
	Sample folder with only six days of data collection.
* **LoED_parser.ipynb**   
	A jupyter notebook to generate statistics of the dataset
* **LoED_parser.py**   
	A python file to generate basic statistics from the dataset
* **LoED_parser.html**  
	HTML rendering of the LoRaDatasetParser.ipynb notebook
