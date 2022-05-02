from django.contrib.auth.models import User
from django.db import models


# Create your models here.
class PayIdMapping(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='pay_id')
    pay_id = models.CharField(max_length=128)

