from django.core.management.base import BaseCommand
from Dog_Breed_Classifier.core.ocr import CharRecognizer
from Dog_Breed_Classifier.models import ImageModel


class Command(BaseCommand):
    def run_extractor(self, image):
        ocr = CharRecognizer(image)
        ocr.run()

    def handle(self, *args, **kwargs):
        non_processed = ImageModel.objects.filter(processed=False)[:2]
        for image in non_processed:
            self.run_extractor(image)
