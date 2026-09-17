# Fixed tests for EMDB XML dataset_info extraction.
import tempfile
import unittest
from pathlib import Path

from dataset_xml import load_dataset_info_from_xml


class DatasetXmlTests(unittest.TestCase):
    def test_load_dataset_info_from_xml_extracts_required_fields(self):
        xml = """<emd emdb_id="EMD-6287">
  <admin><title>T20S reconstruction</title></admin>
  <crossreferences><auxiliary_link_list><auxiliary_link>
    <link>http://dx.doi.org/10.6019/EMPIAR-10025</link>
  </auxiliary_link></auxiliary_link_list></crossreferences>
  <sample>
    <supramolecule_list><sample_supramolecule>
      <oligomeric_state>D7</oligomeric_state>
    </sample_supramolecule></supramolecule_list>
    <macromolecule_list><protein_or_peptide><name>20S proteasome</name></protein_or_peptide></macromolecule_list>
  </sample>
  <structure_determination_list><structure_determination>
    <microscopy_list><single_particle_microscopy>
      <acceleration_voltage>300</acceleration_voltage>
      <nominal_cs>2.7</nominal_cs>
      <image_recording_list><image_recording>
        <average_electron_dose_per_image>53</average_electron_dose_per_image>
      </image_recording></image_recording_list>
    </single_particle_microscopy></microscopy_list>
  </structure_determination></structure_determination_list>
  <map><pixel_spacing><x>0.982</x></pixel_spacing></map>
</emd>"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "emd.xml"
            path.write_text(xml)

            result = load_dataset_info_from_xml(path)

        self.assertEqual(result["emdb_id"], "EMD-6287")
        self.assertEqual(result["empiar_id"], "EMPIAR-10025")
        self.assertEqual(result["pixel_size_A"], 0.982)
        self.assertEqual(result["accelerating_voltage_kv"], 300)
        self.assertEqual(result["spherical_aberration_mm"], 2.7)
        self.assertEqual(result["total_exposure_dose_e_per_A2"], 53)
        self.assertEqual(result["symmetry"], "D7")


if __name__ == "__main__":
    unittest.main()
