package main

/*
#include <stdlib.h>
*/
import "C"
import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"unsafe"

	"github.com/google/go-containerregistry/pkg/authn"
	"github.com/google/go-containerregistry/pkg/name"
	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/remote"
)

// OCIError represents an error that occurred during OCI operations
type OCIError struct {
	Code    int
	Message string
}

// OCIResult represents the result of pulling a model
type OCIResult struct {
	LocalPath string
	Digest    string
	Error     *OCIError
}

//export PullOCIModel
func PullOCIModel(imageRef, cacheDir *C.char) *C.char {
	goImageRef := C.GoString(imageRef)
	goCacheDir := C.GoString(cacheDir)

	result, err := pullModel(goImageRef, goCacheDir)
	if err != nil {
		if result == nil {
			result = &OCIResult{}
		}
		result.Error = &OCIError{
			Code:    1,
			Message: err.Error(),
		}
	}

	jsonBytes, _ := json.Marshal(result)
	return C.CString(string(jsonBytes))
}

//export FreeString
func FreeString(s *C.char) {
	C.free(unsafe.Pointer(s))
}

func pullModel(imageRef, cacheDir string) (*OCIResult, error) {
	ctx := context.Background()

	// Parse the image reference
	ref, err := name.ParseReference(imageRef)
	if err != nil {
		return nil, fmt.Errorf("failed to parse image reference: %w", err)
	}

	// Use docker config for authentication (supports docker login)
	authenticator := authn.NewMultiKeychain(
		authn.DefaultKeychain,
	)

	// Get the image descriptor
	img, err := remote.Image(ref, remote.WithAuthFromKeychain(authenticator), remote.WithContext(ctx))
	if err != nil {
		return nil, fmt.Errorf("failed to fetch image: %w", err)
	}

	// Get the manifest
	manifest, err := img.Manifest()
	if err != nil {
		return nil, fmt.Errorf("failed to get manifest: %w", err)
	}

	// Find the GGUF layer
	var ggufLayer v1.Layer
	var ggufDigest string
	for _, layer := range manifest.Layers {
		mediaType := string(layer.MediaType)
		if mediaType == "application/vnd.docker.ai.gguf.v3" || strings.Contains(mediaType, "gguf") {
			ggufLayer, err = img.LayerByDigest(layer.Digest)
			if err != nil {
				return nil, fmt.Errorf("failed to get GGUF layer: %w", err)
			}
			ggufDigest = layer.Digest.String()
			break
		}
	}

	if ggufLayer == nil {
		return nil, fmt.Errorf("no GGUF layer found in image")
	}

	// Prepare local file path
	refStr := ref.String()
	modelFilename := strings.ReplaceAll(refStr, "/", "_")
	modelFilename = strings.ReplaceAll(modelFilename, ":", "_")
	modelFilename += ".gguf"

	localPath := filepath.Join(cacheDir, modelFilename)

	// Check if file already exists
	if _, err := os.Stat(localPath); err == nil {
		// File exists, verify digest matches
		return &OCIResult{
			LocalPath: localPath,
			Digest:    ggufDigest,
		}, nil
	}

	// Download the layer
	layerReader, err := ggufLayer.Uncompressed()
	if err != nil {
		return nil, fmt.Errorf("failed to get layer reader: %w", err)
	}
	defer layerReader.Close()

	// Create the local file
	outFile, err := os.Create(localPath + ".tmp")
	if err != nil {
		return nil, fmt.Errorf("failed to create output file: %w", err)
	}

	// Copy the data
	_, err = io.Copy(outFile, layerReader)
	outFile.Close()
	if err != nil {
		os.Remove(localPath + ".tmp")
		return nil, fmt.Errorf("failed to write layer data: %w", err)
	}

	// Rename to final location (atomic operation)
	if err := os.Rename(localPath+".tmp", localPath); err != nil {
		os.Remove(localPath + ".tmp")
		return nil, fmt.Errorf("failed to rename file: %w", err)
	}

	return &OCIResult{
		LocalPath: localPath,
		Digest:    ggufDigest,
	}, nil
}

func main() {}
