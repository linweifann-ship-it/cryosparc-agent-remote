import unittest

from cryosparc_agent_remote.vision_inputs import (
    MAX_VISUAL_CONTACT_SHEET_PIXELS,
    bounded_visual_layout,
)


class VisionContextBudgetTests(unittest.TestCase):
    def test_large_class_sheet_is_bounded_without_changing_item_count(self):
        count, tile_size, columns, rows = bounded_visual_layout(50, 192, 5, 26, 50)

        self.assertEqual((count, columns, rows), (50, 5, 10))
        self.assertLess(tile_size, 192)
        self.assertLessEqual(columns * tile_size * rows * (tile_size + 26), MAX_VISUAL_CONTACT_SHEET_PIXELS)

    def test_micrograph_request_keeps_even_sampling_count_but_bounds_resolution(self):
        count, tile_size, columns, rows = bounded_visual_layout(6, 512, 3, 42, 6)

        self.assertEqual((count, columns, rows), (6, 3, 2))
        self.assertLess(tile_size, 512)
        self.assertLessEqual(columns * tile_size * rows * (tile_size + 42), MAX_VISUAL_CONTACT_SHEET_PIXELS)
