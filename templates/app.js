let currentFilename = null;
        let numSlices = 1;
        let currentSlice = 0;
        let scale = 1;
        let translateX = 0;
        let translateY = 0;
        let rotation = 0;
        let isPanning = false;
        let isAnnotating = false;
        let isDrawing = false;
        let startX, startY;
        let isDicom = false;

        const preview = document.getElementById('preview');
        const placeholder = document.getElementById('placeholder');
        const annotationCanvas = document.getElementById('annotationCanvas');
        const ctx = annotationCanvas.getContext('2d');
        const sliceSlider = document.getElementById('sliceSlider');
        const sliceNumber = document.getElementById('sliceNumber');
        const totalSlices = document.getElementById('totalSlices');
        const sliceControls = document.getElementById('sliceControls');
        const sliceNumberDisplay = document.getElementById('sliceNumberDisplay');
        const sliceButtonsTop = document.getElementById('sliceButtonsTop');
        const sliceButtonsBottom = document.getElementById('sliceButtonsBottom');
        const prevSliceTopBtn = document.getElementById('prevSliceTop');
        const nextSliceTopBtn = document.getElementById('nextSliceTop');
        const prevSliceBottomBtn = document.getElementById('prevSliceBottom');
        const nextSliceBottomBtn = document.getElementById('nextSliceBottom');
        const imageContainer = document.getElementById('imageContainer');

        // Update image and canvas transform
        function updateTransform() {
            const transform = `translate(-50%, -50%) translate(${translateX}px, ${translateY}px) scale(${scale}) rotate(${rotation}deg)`;
            preview.style.transform = transform;
            annotationCanvas.style.transform = transform;
        }

        // Resize canvas to match image
        function resizeCanvas() {
            const container = document.getElementById('imageContainer');
            annotationCanvas.width = container.clientWidth - 12; // Account for padding
            annotationCanvas.height = container.clientHeight - 12;
            clearAnnotations();
        }

        // Clear annotations
        function clearAnnotations() {
            ctx.clearRect(0, 0, annotationCanvas.width, annotationCanvas.height);
        }

        // Load new slice
        function loadSlice(sliceIndex) {
            if (sliceIndex < 0 || sliceIndex >= numSlices) return;
            currentSlice = sliceIndex;
            sliceSlider.value = numSlices - 1 - sliceIndex; // Reverse for vertical
            sliceNumber.textContent = sliceIndex + 1;
            preview.src = `/get_slice/${currentFilename}/${sliceIndex}`;
            clearAnnotations();
            updateTransform();
        }

        // Reset transformations
        function resetTransformations() {
            scale = 1;
            translateX = 0;
            translateY = 0;
            rotation = 0;
            isPanning = false;
            isAnnotating = false;
            preview.style.cursor = 'move';
            annotationCanvas.style.cursor = 'move';
            document.getElementById('togglePan').textContent = 'Toggle Pan';
            document.getElementById('toggleAnnotate').textContent = 'Toggle Annotate';
            clearAnnotations();
            updateTransform();
        }

        // File upload handler
        document.getElementById('dicomFile').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (!file) return;

            const formData = new FormData();
            formData.append('file', file);

            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) throw new Error('Upload failed');
                return response.json();
            })
            .then(data => {
                if (data.error) throw new Error(data.error);

                // Reset transformations and show image
                resetTransformations();
                placeholder.classList.add('hidden');
                preview.src = data.initial_image;
                preview.classList.remove('hidden');
                annotationCanvas.classList.remove('hidden');
                imageContainer.style.backgroundColor = '#000000'; // Black background
                resizeCanvas();

                // Set up slice controls
                numSlices = data.num_slices;
                isDicom = data.is_dicom;
                currentFilename = data.initial_image.split('/')[2];
                currentSlice = 0;

                totalSlices.textContent = numSlices;
                sliceNumber.textContent = 1;
                sliceSlider.max = numSlices - 1;
                sliceSlider.value = numSlices - 1; // Top position = slice 0
                sliceControls.classList.toggle('hidden', !isDicom || numSlices === 1);
                sliceNumberDisplay.classList.toggle('hidden', !isDicom || numSlices === 1);
                sliceButtonsTop.classList.toggle('hidden', !isDicom || numSlices === 1);
                sliceButtonsBottom.classList.toggle('hidden', !isDicom || numSlices === 1);
            })
            .catch(error => {
                alert('Error: ' + error.message);
            });
        });

        // Slice navigation
        sliceSlider.addEventListener('input', function(e) {
            const invertedIndex = numSlices - 1 - parseInt(e.target.value);
            loadSlice(invertedIndex);
        });

        prevSliceTopBtn.addEventListener('click', () => {
            if (currentSlice > 0) loadSlice(currentSlice - 1);
        });

        nextSliceTopBtn.addEventListener('click', () => {
            if (currentSlice < numSlices - 1) loadSlice(currentSlice + 1);
        });

        prevSliceBottomBtn.addEventListener('click', () => {
            if (currentSlice > 0) loadSlice(currentSlice - 1);
        });

        nextSliceBottomBtn.addEventListener('click', () => {
            if (currentSlice < numSlices - 1) loadSlice(currentSlice + 1);
        });

        // Scroll to change slices
        function handleWheel(e) {
            if (!isDicom || numSlices === 1) return;
            e.preventDefault();
            const delta = Math.sign(e.deltaY);
            const newSlice = currentSlice + delta;
            loadSlice(newSlice);
        }

        preview.addEventListener('wheel', handleWheel);
        annotationCanvas.addEventListener('wheel', handleWheel);

        // Zoom controls
        document.getElementById('zoomIn').addEventListener('click', () => {
            scale = Math.min(scale + 0.2, 3);
            updateTransform();
        });

        document.getElementById('zoomOut').addEventListener('click', () => {
            scale = Math.max(scale - 0.2, 0.5);
            updateTransform();
        });

        // Pan controls
        document.getElementById('togglePan').addEventListener('click', () => {
            isPanning = !isPanning;
            isAnnotating = false;
            preview.style.cursor = isPanning ? 'grab' : 'move';
            annotationCanvas.style.cursor = isPanning ? 'grab' : 'move';
            document.getElementById('togglePan').textContent = isPanning ? 'Disable Pan' : 'Toggle Pan';
            document.getElementById('toggleAnnotate').textContent = 'Toggle Annotate';
            updateTransform();
        });

        preview.addEventListener('mousedown', (e) => {
            if (isPanning) {
                e.preventDefault();
                startX = e.clientX - translateX;
                startY = e.clientY - translateY;
                preview.style.cursor = 'grabbing';
                annotationCanvas.style.cursor = 'grabbing';
                const onMouseMovePan = (e) => {
                    translateX = e.clientX - startX;
                    translateY = e.clientY - startY;
                    updateTransform();
                };
                preview.addEventListener('mousemove', onMouseMovePan);
                const stopPanning = () => {
                    preview.removeEventListener('mousemove', onMouseMovePan);
                    if (isPanning) {
                        preview.style.cursor = 'grab';
                        annotationCanvas.style.cursor = 'grab';
                    }
                };
                preview.addEventListener('mouseup', stopPanning, { once: true });
                preview.addEventListener('mouseleave', stopPanning, { once: true });
            }
        });

        // Annotation controls
        document.getElementById('toggleAnnotate').addEventListener('click', () => {
            isAnnotating = !isAnnotating;
            isPanning = false;
            preview.style.cursor = isAnnotating ? 'crosshair' : 'move';
            annotationCanvas.style.cursor = isAnnotating ? 'crosshair' : 'move';
            document.getElementById('toggleAnnotate').textContent = isAnnotating ? 'Disable Annotate' : 'Toggle Annotate';
            document.getElementById('togglePan').textContent = 'Toggle Pan';
            updateTransform();
        });

        annotationCanvas.addEventListener('mousedown', (e) => {
            if (!isAnnotating) return;
            e.preventDefault();
            isDrawing = true;
            const rect = annotationCanvas.getBoundingClientRect();
            ctx.beginPath();
            ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
            ctx.strokeStyle = 'red';
            ctx.lineWidth = 2;
            ctx.lineCap = 'round';
        });

        annotationCanvas.addEventListener('mousemove', (e) => {
            if (!isDrawing || !isAnnotating) return;
            const rect = annotationCanvas.getBoundingClientRect();
            ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
            ctx.stroke();
        });

        annotationCanvas.addEventListener('mouseup', () => {
            isDrawing = false;
            ctx.closePath();
        });

        annotationCanvas.addEventListener('mouseleave', () => {
            isDrawing = false;
            ctx.closePath();
        });

        // Rotate control
        document.getElementById('rotate').addEventListener('click', () => {
            rotation = (rotation + 90) % 360;
            updateTransform();
        });

        // Reset control
        document.getElementById('reset').addEventListener('click', () => {
            resetTransformations();
        });

        // Resize canvas on window resize
        window.addEventListener('resize', resizeCanvas);