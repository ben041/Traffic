# Generated by Django 5.0.7 on 2024-11-27 22:30

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0002_area_vehicle_is_suspected_vehicle_reason_suspected_and_more'),
    ]

    operations = [
        migrations.RenameField(
            model_name='area',
            old_name='camera_feed_url',
            new_name='video_url',
        ),
        migrations.AddField(
            model_name='area',
            name='use_video_file',
            field=models.BooleanField(default=True),
        ),
    ]
