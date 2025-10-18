Error & Warning
===

Madd records errors and warnings when running. But for memory saving, only the last several error and warning items saved. You can check the macro `MADD_ERROR_MAX` in basic/basic.h, and you can also modify it before building Madd.

Output Error & Warning
---

You can call the following functions to print the last error and warning item(s) at any time. And you can also write the information to file.

```C
// print
void Madd_Error_Print_Last(void);
void Madd_Error_Print_All(void);

// save
void Madd_Error_Save_Last(FILE *fp);
void Madd_Error_Save_All(FILE *fp);
```

You can also set a log file to save the output instantly.

```C
bool Madd_Error_Enable_Logfile(const char *log_file_name);
void Madd_Error_Set_Logfile(FILE *fp);
void Madd_Error_Close_Logfile(void);
```

Get Last Item Manually
---

If you already call the `Madd_Error_Enable_Logfile` and then call `Madd_Error_Set_Logfile`, the file from `Madd_Error_Enable_Logfile` will be closed and Madd will detour to the new FILE in `Madd_Error_Set_Logfile`.

If you are confident to manage the error item by yourself, you can call the following function to get the last item. `Madd_Error_Item` is a `struct` to save the error/warning item. This function will copy the last item, so you don't need to worry that you may change the error/warning records of Madd. The function return `MADD_SUCCESS` if there is no error or warning, or return the sign of the last item (return `MADD_ERROR` if error, `MADD_WARNING` if warning).

```C
char Madd_Error_Get_Last(Madd_Error_Item *mei);
```

*Note: all strings in Madd Error are wide characters.*

Global Variables
---

* `madd_error_keep_print` if `true`, Madd keeps print the error/warning item at any time it pops.
* `madd_error_color_print` if `true`, Madd will print the error/warning with colors. This feature is based on the ESC (escape character), which may not be supported on some platforms like Windows 8 or earlier.
* `madd_error_exit` if `true`, Madd will exit immediately when an *error* is popped. I **strongly recommend** you to set it `true` in your main function, because Madd won't guarantee the correctness after an error.
* `madd_warning_exit` if `true`, Madd will exit immediately when a *warning* is popped.
* `madd_error_n` number of error & warning items.
* `madd_error.n_error` & `madd_error.n_warning` number of error / warning items.