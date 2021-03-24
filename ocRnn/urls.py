from rest_framework.routers import DefaultRouter

from .views import ImageModelViewSet

router = DefaultRouter()
router.register(r'ocRnn', ImageModelViewSet)
urlpatterns = router.urls
