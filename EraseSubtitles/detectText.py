from text_detection.predict import get_text_boxes
import cv2

def resize_img(image):
    h, w = image.shape[:2]
    rescale_fac = max(h, w) / 1000
    if rescale_fac > 1.0:
        h = int(h / rescale_fac)
        w = int(w / rescale_fac)
    return h, w, rescale_fac

def get_coords(num_of_frames, masks, sample_step=5):
    xmin, ymin, xmax, ymax = 10000, 10000, 0, 0
    new_coords = [xmin, ymin, xmax, ymax]
    rh, rw, rescale_fac = resize_img(masks[0])
    max_height, max_width = masks[0].shape[:2]
    print('Original Dimensions: ', max_height, 'x', max_width)
    print('Rescaled Dimensions: ', rh, 'x', rw)

    # Sample every Nth frame — subtitle regions are static, so we don't need every frame
    sampled_indices = list(range(0, num_of_frames, sample_step))
    print(f'Sampling {len(sampled_indices)} of {num_of_frames} frames (step={sample_step})')

    for idx, i in enumerate(sampled_indices):
        if idx % 50 == 0:
            print(f'  CRAFT text detection: frame {i}/{num_of_frames} ({idx}/{len(sampled_indices)} sampled)')
        input_img = cv2.resize(masks[i], (rw,rh))
        text = get_text_boxes(input_img)

        for coord in text:
            x = coord[::2]
            y = coord[1::2]
            xmin = min(x)
            ymin = min(y)
            xmax = max(x)
            ymax = max(y)
        
        if new_coords[0] > xmin:
            new_coords[0] = xmin
        
        if new_coords[1] > ymin:
            new_coords[1] = ymin

        if new_coords[2] < xmax:
            new_coords[2] = xmax
        
        if new_coords[3] < ymax:
            new_coords[3] = ymax

    new_coords = [int(coord) for coord in new_coords]

    xmin, ymin, xmax, ymax = new_coords

    # Inverse rescaling (getting back to original coordinates)
    xmin = int(xmin * rescale_fac)
    if xmin - 10 >= 0:
      xmin -= 10
      
    ymin = int(ymin * rescale_fac)
    if ymin - 10 >= 0:
      ymin -= 10

    xmax = int(xmax * rescale_fac)
    if xmax + 10 <= max_width:
      xmax += 10

    ymax = int(ymax * rescale_fac)
    if ymax + (20 * rescale_fac) <= max_height:
      ymax += int(20 * rescale_fac)

    # Clamp to valid frame bounds
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(max_width, xmax)
    ymax = min(max_height, ymax)

    # Sanity check: if detected region covers >50% of frame height,
    # the color segmentation is too noisy. Constrain to bottom subtitle area.
    region_height = ymax - ymin
    if region_height > max_height * 0.5:
        print(f'  WARNING: Detected region too large ({region_height}px / {max_height}px height).')
        print(f'  Constraining to bottom 35% of frame (typical subtitle area).')
        ymin = int(max_height * 0.65)
        ymax = max_height
        xmin = 0
        xmax = max_width

    print(f'  Final subtitle region: [{xmin}, {ymin}, {xmax}, {ymax}]')
    return [xmin, ymin, xmax, ymax]

