import unittest

from dataset_info import normalize_dataset_info


class DatasetInfoTests(unittest.TestCase):
    def test_user_facing_acquisition_names_are_normalized(self):
        result = normalize_dataset_info({
            "pixel_size_A": 0.6575,
            "accelerating_voltage_kv": 300,
            "spherical_aberration_mm": 2.7,
            "total_exposure_dose_e_per_A2": 53,
        })
        self.assertEqual(result["psize_A"], 0.6575)
        self.assertEqual(result["accel_kv"], 300)
        self.assertEqual(result["cs_mm"], 2.7)
        self.assertEqual(result["total_dose_e_per_A2"], 53)

    def test_nested_acquisition_parameters_are_supported(self):
        result = normalize_dataset_info({
            "acquisition_parameters": {"pixel_size_A": 0.6575},
        })
        self.assertEqual(result["psize_A"], 0.6575)

    def test_raw_initial_micrograph_inputs_are_normalized(self):
        result = normalize_dataset_info({
            "micrographs_data_path": "/data/micrographs/*.mrc",
            "pixel_size_A": 0.6575,
            "voltage_kV": 300,
            "spherical_aberration_mm": 2.7,
            "total_exposure_dose_e_per_A2": 53,
        })
        self.assertEqual(
            result["available_input_files"]["micrograph_blob_paths"],
            "/data/micrographs/*.mrc",
        )
        self.assertEqual(result["psize_A"], 0.6575)
        self.assertEqual(result["accel_kv"], 300)
        self.assertEqual(result["cs_mm"], 2.7)
        self.assertEqual(result["total_dose_e_per_A2"], 53)

    def test_raw_movies_list_is_normalized_to_a_dictionary_file_context(self):
        result = normalize_dataset_info({"available_input_files": ["bad"], "raw_movies": ["/data/movies/*.tif"]})
        self.assertEqual(result["available_input_files"], {"movie_blob_paths": ["/data/movies/*.tif"]})


if __name__ == "__main__":
    unittest.main()
