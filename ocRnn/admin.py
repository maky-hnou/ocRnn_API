from django.contrib import admin
from django.db.models import Q
from django.utils.translation import gettext_lazy as _
from ocRnn.core.recognition import CharRecognizer

from .models import ImageModel


class InputFilter(admin.SimpleListFilter):
    template = 'admin/input_filter.html'

    def lookups(self, request, model_admin):
        return ((),)

    def choices(self, changelist):
        # Grab only the "all" option.
        all_choice = next(super().choices(changelist))
        all_choice['query_parts'] = (
            (k, v)
            for k, v in changelist.get_filters_params().items()
            if k != self.parameter_name
        )
        yield all_choice


class CategoryFilter(InputFilter):
    parameter_name = 'category'
    title = _('category')

    def queryset(self, request, queryset):
        if self.value() is not None:
            category = self.value()
            return queryset.filter(
                Q(category=category)
            )


class ImageModelAdmin(admin.ModelAdmin):
    search_fields = [field.name for field in ImageModel._meta.get_fields()]
    list_display = [field.name for field in ImageModel._meta.get_fields()]
    list_filter = ('processed', 'upload_date', CategoryFilter,)
    actions = ('run_classification', )

    def run_classifier(self, image):
        classifier = CharRecognizer(image)
        classifier.get_dog_category()

    def run_classification(self, *args, **kwargs):
        non_processed = ImageModel.objects.filter(processed=False)[:2]
        for image in non_processed:
            self.run_classifier(image)


admin.site.register(ImageModel, ImageModelAdmin)
