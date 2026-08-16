export function fileToBase64(file: File): Promise<string> {
  const MAX_FILE_BYTES = 25 * 1024 * 1024
  if (file.size > MAX_FILE_BYTES) {
    return Promise.reject(new Error('单个文件不能超过 25 MB。'))
  }
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result).split(',')[1])
    reader.onerror = () => reject(reader.error)
    reader.readAsDataURL(file)
  })
}

export function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as string)
    reader.onerror = () => reject(new Error('读取图片失败'))
    reader.readAsDataURL(file)
  })
}
