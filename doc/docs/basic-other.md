Other
===

The minor and unclassified fundamental functions are listed here.

File Read & Write
---

The following functions help you read & save the binary data of your file, assuming you had confirmed the endian type of your saved data. The suffix LE (little endian) and BE (big endian) means the endian type of your **file's data**, not the endian type of your platform. If your data file is of little endian, you should call the functions whose names end with _LE.

```C
union _union8 Read_1byte(FILE *fp);
union _union16 Read_2byte_LE(FILE *fp);
union _union16 Read_2byte_BE(FILE *fp);
union _union32 Read_4byte_LE(FILE *fp);
union _union32 Read_4byte_BE(FILE *fp);
union _union64 Read_8byte_LE(FILE *fp);
union _union64 Read_8byte_BE(FILE *fp);
void Write_1byte(FILE *fp, void *unit);
void Write_2byte_LE(FILE *fp , void *unit);
void Write_2byte_BE(FILE *fp , void *unit);
void Write_4byte_LE(FILE *fp , void *unit);
void Write_4byte_BE(FILE *fp , void *unit);
void Write_8byte_LE(FILE *fp , void *unit);
void Write_8byte_BE(FILE *fp , void *unit);
void Read_Array_LE(FILE *fp, void *buf_, size_t n_element, size_t element_size);
void Read_Array_BE(FILE *fp, void *buf_, size_t n_element, size_t element_size);
void Write_Array_LE(FILE *fp, void *buf_, size_t n_element, size_t element_size);
void Write_Array_BE(FILE *fp, void *buf_, size_t n_element, size_t element_size);
```

To read & save more efficienty, you can set the size of these functions buffer by setting the global variables `madd_file_endian_buf_length`. The larger it is, the faster when you read & write a larget data file.

Print & Save
---

```C
void Madd_Print(wchar_t *wstr);
void Madd_Save(FILE *fp, wchar_t *wstr);
```

All print function in Madd library are calling `Madd_Print` (only 1 parameter is accepted). This function is more fundamental than Madd's error & warning processing. Therefore, any error from `Madd_Print` will result in exiting, even if you set `madd_error_exit = true`.

One important thing is that C standard treats calling `printf` and `wprintf` in a program at the same time is *undefined*. In other words, on some platform, if you called `printf` first, then your next `wprintf` wouldn't work. Thus, Madd provides 2 global variables to indicate the print & save function should output the wide string or narrow string. If `false`, `Madd_Print`/`Madd_Save` transfer the `wstr` to narrow string internally and then call `printf`/`fprintf`. These 2 variables are default `false`, so you can safely use `printf` and `fprintf` in your program.

```C
bool madd_print_wide, madd_save_wide;
```

Time String
---

You can call the time function from time.h. But if you want a simple way to get a time string.

```C
void Time_Stamp_String(time_t t, wchar_t *str);
```