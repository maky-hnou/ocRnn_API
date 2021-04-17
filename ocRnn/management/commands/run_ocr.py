from django.core.management.base import BaseCommand
from ocRnn.core.ocr import CharRecognizer
from ocRnn.models import ImageModel


class Command(BaseCommand):
    def run_extractor(self, image):
        ocr = CharRecognizer(image)
        ocr.run()

    def handle(self, *args, **kwargs):
        non_processed = ImageModel.objects.filter(processed=False)[:2]
        for image in non_processed:
            self.run_extractor(image)
