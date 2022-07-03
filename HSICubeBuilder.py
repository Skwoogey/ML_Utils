import numpy as np
import random
import math
import scipy.ndimage as ndi
import scipy.io as scp
import os
#    a[0:2, 0:2]
#    [[[0, 0], [1, 1]], [[0, 1], [0, 1]]]

class ExtendedNPArray:
    def __init__(self, array):
        self.array = array

    def reformInt(self, key, axis):
        #print('int index', key)
        new_index = max(0, min(self.array.shape[axis] - 1, key))
        #print('new int index', new_index)
        return new_index

    def reformSlicesOrInts(self, key, axis):
        #print('reform splice or int')
        if np.issubdtype(type(key), np.integer):
            return self.reformInt(key, axis)
        elif type(key) == slice:
            #print('slice index', key)
            end_index = key.stop if key.stop != None else self.array.shape[axis]
            start_index = key.start if key.start != None else 0
            list_len = end_index - start_index
            indecies_list = [max(0, min(self.array.shape[axis] - 1, x)) for x in range(start_index, end_index)]
            #print('new slice index', indecies_list)
            return indecies_list
        else:
            raise ValueError("not slice or int")


    def reformArrayOrInt(self, key, axis):
        #print('reform array or int', type(key))
        if np.issubdtype(type(key), np.integer):
            return self.reformInt(key, axis)
        elif type(key) in [list, np.ndarray]:
            new_key = []
            for elem in key:
                new_key.append(self.reformArrayOrInt(elem, axis))
            return new_key
        else:
            raise ValueError("not array or int")


    def __getitem__(self, key):
        #print(type(key))
        #print(key)
        if type(key) == tuple:
            #print('tuple index')
            key = list(key)
        elif type(key) != list:
            key = [key]

        types = [type(x) for x in key]

        if all([x in [list,np.ndarray] or np.issubdtype(type(x), np.integer) for x in types]):
            new_key = []
            for axis_i in range(len(key)):
                new_key.append(self.reformArrayOrInt(key[axis_i], axis_i))
            x = self.array[tuple(new_key)]
            #print(type(x))
            return x
        elif all([x == slice or np.issubdtype(type(x), np.integer) for x in types]):
            index_base = []
            result = self.array
            for axis_i in range(len(key)):
                #print('print me?', self.reformSlicesOrInts(key[axis_i], axis_i))
                result = result[index_base + [self.reformSlicesOrInts(key[axis_i], axis_i)]] 
                index_base += [slice(None)]
            return result
        else:
            raise ValueError('invalid index')

    def __str__(self):
        return self.array.__str__()

class HSIPixelCubeVariation:
    def __init__(self, cube, augment, handler):
        self.handler = handler
        self.variations = [None] * 8
        self.variations[0] = cube
        self.augmented = False
        if augment:
            self.augment()

    def augment(self):
        if self.augmented:
            return
        self.variations[1] = np.flip(self.variations[0], axis = self.handler.axes_order.find("H"))
        self.variations[2] = np.rot90(self.variations[0], 1, [self.handler.axes_order.find("H"), self.handler.axes_order.find("W")])
        self.variations[3] = np.rot90(self.variations[0], 2, [self.handler.axes_order.find("H"), self.handler.axes_order.find("W")])
        self.variations[4] = np.rot90(self.variations[0], 3, [self.handler.axes_order.find("H"), self.handler.axes_order.find("W")])
        self.variations[5] = np.flip(self.variations[2], axis = self.handler.axes_order.find("H"))
        self.variations[6] = np.flip(self.variations[3], axis = self.handler.axes_order.find("H"))
        self.variations[7] = np.flip(self.variations[4], axis = self.handler.axes_order.find("H"))
        self.augmented = True

    def getOriginal(self):
        return [self.variations[0]]

    def getAll(self):
        self.augment()
        return self.variations

class CoordSys:
    def __init__(self, right, down=None):
        self.right = np.array(right, dtype = int)
        if down == None:
            self.down = np.array([self.right[1], -self.right[0]], dtype = int)
        else:
            self.down = np.array(down, dtype = int)

defaultCS = CoordSys([0, 1])

class paddingImage:
    def __init__(self, array):
        self.array = array
        self.center_offset = np.array([0, 0], dtype = int)
        self.total_padded = 0
        self.related_views = []

    def expand(self, border_width):

        self.total_padded += border_width
        print('expanding:', border_width, self.total_padded)
        print(self.array.shape)

        pad_width = [
            (border_width, border_width),
            (border_width, border_width)
        ] + [(0,0)] * (len(self.array.shape) - 2)



        self.array = np.pad(
            self.array, 
            pad_width,
            'edge'
        )

        self.center_offset += border_width
        for rv in self.related_views:
            rv.rebuild_view()

    def addView(self, view):
        if view in self.related_views:
            assert("View has already been added")

        self.related_views.append(view)

class HSICubeImageViews:
    def __init__(self, handler, coord_sys):
        self.handler = handler
        self.image = handler.img
        self.image.addView(self)
        self.coord_sys = coord_sys
        
        self.rebuild_offsets()
        self.rebuild_view()
    
    def rebuild_offsets(self):
        half_window_size = (self.handler.window_size - 1) // 2
        
        d = self.coord_sys.down
        r = self.coord_sys.right
        self.max_radius = np.max([np.absolute(d + r), np.absolute(d - r)]) * half_window_size
        #print(self.max_radius)
        self.center_offset = -(self.coord_sys.down + self.coord_sys.right) * half_window_size
        

    def rebuild_view(self):
        if self.image.total_padded < self.max_radius:
            self.image.expand(self.max_radius - self.image.total_padded)
            return
        #calculating destinaation strides
        src_st = self.image.array.strides
        src_st_vec = np.array([src_st[0], src_st[1]])
        dst_st_vec = tuple([
            np.dot(self.coord_sys.down, src_st_vec),
            np.dot(self.coord_sys.right, src_st_vec)
        ])

        #calculating destination shapes
        src_sh = self.image.array.shape
        
        dst_st = src_st[:2]
        dst_sh = src_sh[:2]
        for axes in self.handler.axes_order:
            if axes == "H":
                dst_st += (dst_st_vec[0],)
                dst_sh += (self.handler.window_size,)
            elif axes == "W":
                dst_st += (dst_st_vec[1],)
                dst_sh += (self.handler.window_size,)
            else:
                index = self.handler.axes_indecies[axes]
                dst_st += (src_st[index],)
                dst_sh += (src_sh[index],)
            

        #print('source shape:', src_sh)
        #print('destination shape:', dst_sh)
        #print('source strides:', src_st)
        #print('destination strides:', dst_st)

        src_arr = self.image.array

        self.views = np.lib.stride_tricks.as_strided(src_arr, dst_sh, dst_st)

    def getHSICube(self, coords):
        shifted_coords = coords + self.center_offset + self.image.center_offset
        #print(coords, self.center_offset, self.image.center_offset)
        #print(shifted_coords)
        assert((shifted_coords >= 0).all())

        return self.views[shifted_coords[0], shifted_coords[1]]

class HSICubeTransformImageExtractor:
    def __init__(self, image, tm, window_size):
        self.image = image
        self.image.addView(self)
        self.window_size = window_size
        self.half_window_size = (window_size - 1) // 2
        
        self.fwd_tm = tm

        self.array_bank = []
        for x0 in range(self.image.array.shape[0]):
            self.array_bank.append([None] * self.image.array.shape[1])

        self.rebuild_view()

    def rebuild_view(self):
        self.inv_tm = np.linalg.inv(self.fwd_tm) 

        corners = np.array([[1.0, 1.0, -1.0, -1.0],
                            [1.0, -1.0, 1.0, -1.0]], dtype=float) * self.half_window_size

        src_corners = np.ceil(np.abs(np.dot(self.inv_tm, corners)))

        self.max_radius = int(np.ceil(np.linalg.norm(np.max(src_corners, axis = 1))))
        self.center_offset = -np.array([self.max_radius, self.max_radius], dtype = int)
        self.diameter = self.max_radius * 2 + 1

        self.map = np.ndarray(shape = (self.window_size, self.window_size, 2))

        for x0 in range(self.window_size):
            for x1 in range(self.window_size):
                oc = np.array([[x0], [x1]], dtype = float)
                self.map[x0, x1] = (np.dot(self.inv_tm, (oc - self.half_window_size)) + self.max_radius).reshape(2)

        if self.image.total_padded < self.max_radius:
            self.image.expand(self.max_radius - self.image.total_padded)
            return

    def getHSICube(self, coords):
        if self.array_bank[coords[0]][coords[1]] != None:
            return self.array_bank[coords[0]][coords[1]]

        #start = time.time()
        sc_start = coords + self.center_offset + self.image.center_offset
        sc_end = sc_start + self.diameter
        src_cube = self.image.array[sc_start[0]:sc_end[0], sc_start[1]:sc_end[1]]

        def transform_func(output_coords):
            #oc = np.array([[output_coords[0]], [output_coords[1]]], dtype = float)

            #ic = np.dot(self.inv_tm, (oc - self.half_window_size)) + self.max_radius
            ic = self.map[output_coords[0], output_coords[1]]
            return (ic[0], ic[1], output_coords[2], output_coords[3])

        dst_cube = ndi.geometric_transform(src_cube, transform_func, 
                    (self.window_size, self.window_size, src_cube.shape[2], src_cube.shape[3]))

        #stop = time.time()
        #print('time:', stop - start)
        #src_cube_center = src_cube[self.max_radius, self.max_radius]
        #dst_cube_center = dst_cube[self.half_window_size, self.half_window_size]

        #distance = np.linalg.norm(src_cube_center - dst_cube_center)
        #print('center distance:', distance)

        self.array_bank[coords[0]][coords[1]] = dst_cube
        #print(coords, self.center_offset, self.image.center_offset)
        #print(shifted_coords)
        #print(dst_cube.shape)
        return dst_cube


    def addTransform(tm, nt):
        return np.dot(tm, nt)

    def getIdentity():
        return np.array([[1.0, 0.0],
                         [0.0, 1.0]], dtype=float)

    def addRotation(tm, a):
        rm = np.array([[np.cos(a), np.sin(a)],
                     [-np.sin(a), np.cos(a)]], dtype = float)

        return HSICubeTransformImageExtractor.addTransform(tm, rm)

    def addScale(tm, s0, s1):
        sm = np.array([[s0, 0.0],
                       [0.0, s1]], dtype = float)

        return HSICubeTransformImageExtractor.addTransform(tm, sm)

    def addSheer(tm, s0, s1):
        sm = np.array([[1.0, s0],
                       [s1, 1.0]], dtype = float)

        return HSICubeTransformImageExtractor.addTransform(tm, sm)

class HSICubeSavedTransforms:
    def __init__(self, path, index_map):
        self.index_map = index_map

        #path = base_folder + os.path.sep + str(window_size) + os.path.sep + name

        self.cubes = scp.loadmat(path)['data']
        
        self.rebuild_view()

    def rebuild_view(self):
        pass

    def getHSICube(self, coords):
        assert(self.index_map[coords[0]][coords[1]] != None)

        index = self.index_map[coords[0]][coords[1]]

        return self.cubes[index]


class HSIPixelCube:
    def __init__(self, coordinates, label, handler):
        self.label = label
        self.handler = handler
        self.coordinates = coordinates
        self.cubes = dict()

    def addCube(self, cube, name, augment = False):
        variations = HSIPixelCubeVariation(cube, augment, self.handler)
        self.cubes[name] = variations

    def getOriginal(self, getLabels = False):
        variation = self.cubes[defaultCS].getOriginal()
        #print(type(variation[0]))
        if not getLabels:
            return variation
        labels = [self.label]
        return variation, labels

    def getAll(self, getLabels = False):
        all_variants = list(self.cubes.values())
        #print(all_variants)
        all_variants_list = []
        for v in all_variants:
            all_variants_list = all_variants_list + v.getAll()
        if not getLabels:
            return all_variants_list    
        labels = [self.label] * len(all_variants_list)
        return all_variants_list, labels

    def get(self, variants, transforms):
        all_variants = list(self.cubes.values())

        if not variants:
            all_variants = [all_variants[0]]
        #print(all_variants)
        all_variants_list = []
        get_func = HSIPixelCubeVariation.getAll if transforms \
                else HSIPixelCubeVariation.getOriginal
        for v in all_variants:
            all_variants_list = all_variants_list + get_func(v)
        
        return all_variants_list    


    def hasVariant(self, variant_name):
        return variant_name in list(self.cubes.keys())

    def clear(self):
        self.cubes = dict()

class HSIImageHandler:
    def __init__(self, hsi_image_file, ground_truth_file, window_size, axes_order="CDHW"):
        hsi_image = np.expand_dims(scp.loadmat(hsi_image_file)['data'], -1)
        ground_truth = scp.loadmat(ground_truth_file)['data']

        self.axes_order = axes_order.upper()
        
        if len(self.axes_order) == 3:
            assert(all([x in self.axes_order for x in 'HWC']))
            assert(all([x in 'HWC' for x in self.axes_order]))
            self.axes_indecies = {
                "C": 2
            }
        elif len(self.axes_order) == 4:
            assert(all([x in self.axes_order for x in 'HWDC']))
            assert(all([x in 'HWDC' for x in self.axes_order]))
            self.axes_indecies = {
                "D": 2,
                "C": 3
            }
        else:
            raise("Axes order must be either 4 or 3 symbols long")
            

        self.img = paddingImage(hsi_image)
        self.gt = ground_truth
        self.pixels = []
        self.labeled_pixels = []
        self.pixels_map = []
        self.linear_map = []
        class_labels, counts = np.unique(self.gt, return_counts=True)
        self.class_count = len(class_labels) -1
        self.pixels_per_class = [[] for _ in range(self.class_count)]
        self.samples_num = []
        self.available_pixels = [None for _ in range(self.class_count)]
        self.variant_views = dict()
        self.variant_keys = None
        self.window_size = window_size
        #self.stp = saved_transforms_path
        print('creating pixel maps...')
        cur_index = 0
        for ax0 in range(self.gt.shape[0]):
            self.pixels_map.append([])
            self.linear_map.append([])
            for ax1 in range(self.gt.shape[1]):
                label = self.gt[ax0, ax1] - 1
                coordinates = np.array([ax0, ax1])
                hpc = HSIPixelCube(coordinates, label, self)
                self.pixels.append(hpc)
                if label != -1:
                    #print(label)
                    self.labeled_pixels.append(hpc)
                    self.pixels_per_class[label].append(hpc)
                    self.linear_map[ax0].append(cur_index)
                    cur_index += 1
                    #print(self.pixels_per_class)
                else:
                    self.linear_map[ax0].append(None)
                self.pixels_map[ax0].append(hpc)

        for i in range(self.class_count):
            self.samples_num.append(len(self.pixels_per_class[i]))
            self.refillClass(i)
        #print(self.samples_num)
        #print(len(self.labeled_pixels))
        print('pixel maps created')

        self.addVariant(defaultCS)
        self.populatePixels(False)

    def createPixelVariant(self, hpc, variant_name, augment):
        cube = self.variant_views[variant_name].getHSICube(hpc.coordinates)
        #print('\r' + str(hpc.coordinates), end='')
        hpc.addCube(cube, variant_name, augment)
        #print(hpc.cubes.keys())

    def populatePixelWithVariants(self, hpc, augment):
        for vn in self.variant_keys:
            if hpc.hasVariant(vn):
                if augment:
                    hpc.cubes[vn].augment()
                continue
            #print(hpc.coordinates, 'populated')
            self.createPixelVariant(hpc, vn, augment)

    def addVariant(self, variant):
        if variant in list(self.variant_views.keys()):
            return
        self.variant_views[variant] = HSICubeImageViews(self, variant)
        self.variant_keys = list(self.variant_views.keys())
        #print(self.variant_windows.keys())
        #print(self.variant_windows[variant])

    def addTransformExtractor(self, tm):
        if tm.tobytes() in list(self.variant_views.keys()):
            return

        self.variant_views[tm.tobytes()] = HSICubeTransformImageExtractor(self.img, tm, self.window_size)
        self.variant_keys = list(self.variant_views.keys())

    def addSavedTransforms(self, path):
        if path in list(self.variant_views.keys()):
            return

        self.variant_views[path] = HSICubeSavedTransforms(path, self.linear_map)
        self.variant_keys = list(self.variant_views.keys())

    def populatePixels(self, augment):
        print('populating pixels...')
        for hpc in self.labeled_pixels:
            self.populatePixelWithVariants(hpc, augment)
        print('pixels populated')

    def getSamplesFromClass(self, class_i, samples_num, augment_variant, augment_transform):
        #print(samples_num, len(self.available_pixels[class_i]))
        
        if samples_num <= len(self.available_pixels[class_i]):
            chosen_pixels = self.available_pixels[class_i][:samples_num]
            self.available_pixels[class_i] = self.available_pixels[class_i][samples_num:]
        else:
            pixels_num = len(self.available_pixels[class_i])
            chosen_pixels = self.available_pixels[class_i]
            self.refillClass(class_i)
            list_i = 0
            while len(chosen_pixels) != samples_num:
                if self.available_pixels[class_i][list_i] not in chosen_pixels:
                    chosen_pixels.append(self.available_pixels[class_i].pop(list_i))
                list_i += 1
        #print("chosen", time.process_time() - start)
        samples = []
        for chosen_pixel in chosen_pixels:
            if augment_variant or augment_transform:
                self.populatePixelWithVariants(chosen_pixel, True)
            samples += chosen_pixel.get(augment_variant, augment_transform)
        #print(pop, got)
        return samples


    def refillClass(self, class_i):
        self.available_pixels[class_i] = self.pixels_per_class[class_i][:]
        random.shuffle(self.available_pixels[class_i])


    def getFullSplit(self, class_sample_nums, augment_variant, augment_transform, one_hot = False):
        #start = time.process_time()
        samples = []
        labels = []
        for class_i in range(self.class_count):
            if class_sample_nums[class_i] != 0:
                samples_from_class = self.getSamplesFromClass(class_i, class_sample_nums[class_i], augment_variant, augment_transform)
                samples += samples_from_class
                labels += [class_i] * len(samples_from_class)

        if one_hot:
            I = np.identity(self.class_count)
            labels = [I[label_i] for label_i in labels]

        #print('finished', time.process_time() - start)
        return samples, labels

    def getRemainingSamples(self, augment_variant=False, augment_transform=False, one_hot = False):
        return self.getFullSplit([len(x) if len(x) != 0 else 1 for x in self.available_pixels], augment_variant, augment_transform, one_hot)

    def getRemainingSamplesAsHSIC(self):
        samples = []
        for class_i in range(self.class_count):
            samples = self.available_pixels[class_i]
            self.available_pixels[class_i] = []

        return samples

    def getPercentageSplit(self, percentage, augment_variant=False, augment_transform=False, one_hot = False):
        return self.getFullSplit([math.ceil(x * percentage) for x in self.samples_num], augment_variant, augment_transform, one_hot)

    def getCountSplit(self, count, augment_variant=False, augment_transform=False, one_hot = False):
        return self.getFullSplit([count] * self.class_count, augment_variant, augment_transform, one_hot)

    def getAll(self, augment_variant=False, augment_transform=False, one_hot=False):
        samples = []
        labels = []
        for i, class_i in enumerate(self.pixels_per_class):
            samples_from_class = []
            if augment:
                for pxl in class_i:
                    #print(pxl.getAll())
                    samples_from_class += pxl.getAll()
            else:
                for pxl in class_i:
                    #print(pxl.getOriginal().shape)
                    samples_from_class += pxl.getOriginal()

            samples += samples_from_class
            labels += [i] * len(samples_from_class)

        if one_hot:
            I = np.identity(self.class_count)
            labels = [I[label_i] for label_i in labels]

        #print('finished', time.process_time() - start)
        return samples, labels

    def clear(self):
        self.variant_views = dict()
        self.variant_keys = None
        for hpc in self.labeled_pixels:
            hpc.clear()



if __name__ == '__main__':
    i = np.arange(200).reshape((10, 10, 2))
    gt = np.random.randint(5, size = (10, 10))
    print(gt)
    HSI_IH = HSIImageHandler(i, gt, 3)
    num_of_classes = HSI_IH.class_count
    HSI_IH.addVariant(CoordSys([1, 1]))
    HSI_IH.addVariant(CoordSys([2, 1]))
    HSI_IH.addVariant(CoordSys([2, -1]))
    HSI_IH.addVariant(CoordSys([2, 2]))
    HSI_IH.addVariant(CoordSys([0, 2]))
    HSI_IH.addVariant(CoordSys([0, 2], [1, 0]))
    HSI_IH.addVariant(CoordSys([0, 1], [2, 0]))



    s, l = HSI_IH.getRemainingSamples(True)
    print(len(s))
    print(l)
    #for x in s:
        #print(x.shape)

