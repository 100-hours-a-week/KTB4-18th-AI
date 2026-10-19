"""사진 입력 검증·정규화를 외부 호출 없이 검증한다."""

import io

import pytest
from fastapi import HTTPException
from PIL import Image

from backend.chat.photo import image


def jpeg():
    output = io.BytesIO()
    photo = Image.new("RGB", (40, 20), "green")
    exif = Image.Exif()
    exif[274] = 6
    exif[270] = "private metadata"
    photo.save(output, "JPEG", exif=exif)
    return output.getvalue()


def png():
    output = io.BytesIO()
    Image.new("RGB", (40, 20), "green").save(output, "PNG")
    return output.getvalue()


def test_jpeg_normalization_strips_metadata_and_corrects_orientation():
    data = image.normalize_image(jpeg())
    with Image.open(io.BytesIO(data)) as result:
        assert result.size == (20, 40)
        assert not result.getexif()


@pytest.mark.parametrize("size", [(5712, 4284), (8064, 6048)])
def test_phone_photo_resolutions_are_accepted_and_downscaled(size):
    output = io.BytesIO()
    with Image.new("RGB", size, "green") as photo:
        photo.save(output, "JPEG")
    data = image.normalize_image(output.getvalue())
    with Image.open(io.BytesIO(data)) as result:
        assert result.size == (1024, 768)


def test_large_photo_is_decoded_at_reduced_scale(monkeypatch):
    output = io.BytesIO()
    Image.new("RGB", (4096, 3072), "green").save(output, "JPEG")
    loaded = []
    original = Image.Image.load

    def load(self):
        loaded.append(self.size)
        return original(self)

    monkeypatch.setattr(Image.Image, "load", load)
    image.normalize_image(output.getvalue())
    # NOTE: draft()가 적용되면 원본(4096×3072)이 아니라 1024px 이상인 축소 크기로 디코딩된다.
    assert loaded[0] == (2048, 1536)


@pytest.mark.parametrize("data,status", [
    (b"not an image", 415),
    (b"\xff\xd8\xffbroken", 415),
    (png(), 415),
    (b"x" * (image.MAX_BYTES + 1), 413),
])
def test_invalid_photo_before_any_model_call(data, status):
    with pytest.raises(HTTPException) as error:
        image.normalize_image(data)
    assert error.value.status_code == status


def test_pixel_limit_uses_original_size_before_reduced_decoding(monkeypatch):
    output = io.BytesIO()
    Image.new("RGB", (4096, 3072), "green").save(output, "JPEG")
    monkeypatch.setattr(image, "MAX_PIXELS", 4096 * 3072 - 1)
    with pytest.raises(HTTPException) as error:
        image.normalize_image(output.getvalue())
    assert error.value.status_code == 413

