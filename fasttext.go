package fasttextgo

// #cgo LDFLAGS: -L${SRCDIR} -lfasttext -lstdc++ -lm
// #include <stdlib.h>
// void load_model(char *name, char *pathZ);
// int predict(char* name, char *query, float *prob, char **buf, int *count, int k, int buf_sz);
// int predictMaxIntention(char* name, char *query, float *prob, char **buf, int *count, int buf_sz);
import "C"
import (
	"errors"
	"unsafe"
)

// LoadModel - load FastText model
func LoadModel(name, path string) {
	p1 := C.CString(name)
	p2 := C.CString(path)

	C.load_model(p1, p2)

	C.free(unsafe.Pointer(p1))
	C.free(unsafe.Pointer(p2))
}

// Predict - predict, return the topN predicted label and their corresponding probability
func Predict(name, sentence string, topN int) (map[string]float32, error) {
	result := make(map[string]float32)

	//add new line to sentence, due to the fasttext assumption
	sentence += "\n"

	cprob := make([]C.float, topN, topN)
	buf := make([]*C.char, topN, topN)
	var resultCnt C.int
	for i := 0; i < topN; i++ {
		buf[i] = (*C.char)(C.calloc(64, 1))
	}

	np := C.CString(name)
	data := C.CString(sentence)

	ret := C.predict(np, data, &cprob[0], &buf[0], &resultCnt, C.int(topN), 64)
	if ret != 0 {
		return result, errors.New("error in prediction")
	} else {
		for i := 0; i < int(resultCnt); i++ {
			result[C.GoString(buf[i])] = float32(cprob[i])
		}
	}
	//free the memory used by C
	C.free(unsafe.Pointer(data))
	C.free(unsafe.Pointer(np))
	for i := 0; i < topN; i++ {
		C.free(unsafe.Pointer(buf[i]))
	}

	return result, nil
}

func PredictMaxIntention(name, sentence string) ([]string, []float32, error) {
	resultLabel := make([]string, 0, 6)
	resultScore := make([]float32, 0, 6)

	//add new line to sentence, due to the fasttext assumption
	sentence += "\n"

	cprob := make([]C.float, 6, 6)
	buf := make([]*C.char, 6, 6)
	var resultCnt C.int
	for i := 0; i < 6; i++ {
		buf[i] = (*C.char)(C.calloc(1280, 1))
	}

	np := C.CString(name)
	data := C.CString(sentence)

	ret := C.predictMaxIntention(np, data, &cprob[0], &buf[0], &resultCnt, 1280)
	if ret != 0 {
		return resultLabel, resultScore, errors.New("error in prediction")
	} else {
		for i := 0; i < int(resultCnt); i++ {
			resultLabel = append(resultLabel, C.GoString(buf[i]))
			resultScore = append(resultScore, float32(cprob[i]))
		}
	}
	//free the memory used by C
	C.free(unsafe.Pointer(data))
	C.free(unsafe.Pointer(np))
	for i := 0; i < 6; i++ {
		C.free(unsafe.Pointer(buf[i]))
	}

	return resultLabel, resultScore, nil
}
