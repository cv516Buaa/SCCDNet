from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class FGSCRDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('background', '001.Nimitz-class_aircraft_carrier', '002.KittyHawk-class_aircraft_carrier',
           '003.Midway-class_aircraft_carrier', '004.Kuznetsov-class_aircraft_carrier',
           '005.Charles_de_Gaulle_aricraft_carrier', '006.INS_Virrat_aircraft_carrier',
           '007.INS_Vikramaditya_aircraft_carrier', '008.Ticonderoga-class_cruiser',
           '009.Arleigh_Burke-class_destroyer', '010.Akizuki-class_destroyer',
           '011.Asagiri-class_destroyer', '012.Kidd-class_destroyer', '013.Type_45_destroyer',
           '014.Wasp-class_assault_ship', '015.Osumi-class_landing_ship', '016.Hyuga-class_helicopter_destroyer',
           '017.Lzumo-class_helicopter_destroyer', '018.Whitby_Island-class_dock_landing_ship',
           '019.San_Antonio-class_transport_dock', '020.Freedom-class_combat_ship',
           '021.Independence-class_combat_ship', '022.Sacramento-class_support_ship', '023.Crane_ship',
           '024.Abukuma-class_frigate', '025.Megayacht', '026.Cargo_ship', '027.Murasame-class_destroyer',
           '028.Container_ship', '029.Towing_vessel', '030.Civil_yacht', '031.Medical_ship', '032.Sand_carrier',
           '033.Tank_ship', '034.Garibaldi_aircraft_carrier', '035.Zumwalt-class_destroyer',
           '036.Kongo-class_destroyer', '037.Horizon-class_destroyer', '038.Atago-class_destroyer',
           '039.Mistral-class_amphibious_assault_ship', '040.Juan_Carlos_I_Strategic_Projection_Ship',
           '041.Maestrale-class_frigate', '042.San_Giorgio-class_transport_dock'),
        palette=[[0, 0, 0], [0, 0, 63], [0, 0, 126], [0, 0, 189], [0, 63, 0], [0, 63, 63], [0, 63, 126],
                [0, 63, 189], [0, 126, 0], [0, 126, 63], [0, 126, 126], [0, 126, 189], [0, 189, 0], [0, 189, 63],
                [0, 189, 126], [0, 189, 189], [63, 0, 0], [63, 0, 63], [63, 0, 126], [63, 0, 189], [63, 63, 0],
                [63, 63, 63], [63, 63, 126], [63, 63, 189], [63, 126, 0], [63, 126, 63], [63, 126, 126], [63, 126, 189],
                [63, 189, 0], [63, 189, 63], [63, 189, 126], [63, 189, 189], [126, 0, 0], [126, 0, 63], [126, 0, 126],
                [126, 0, 189], [126, 63, 0], [126, 63, 63], [126, 63, 126], [126, 63, 189], [126, 126, 0], [126, 126, 63],
                 [126, 126, 126]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
