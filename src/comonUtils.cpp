#ifdef WIN32 // windows
#include <direct.h>
#include <windows.h>
#else
#include <unistd.h>
#include <limits.h>
#endif // linux

#include "comonUtils.h"


/* returns the working directory from where the program is called
https://www.tutorialspoint.com/find-out-the-current-working-directory-in-c-cplusplus
*/
std::string getWorkingDir() {
#ifdef WIN32 // windows
	char buffer[MAX_PATH]; //create string buffer to hold path
	_getcwd(buffer, MAX_PATH);
#else // linux
	char buffer[PATH_MAX];
	getcwd(buffer, PATH_MAX);
#endif
	std::string current_working_dir(buffer);
	return current_working_dir;
}

/* return the directory where this program is located
* http://www.cplusplus.com/forum/general/11104/
* https://stackoverflow.com/questions/875249/how-to-get-current-directory
*/
std::string getProgramDir()
{
#ifdef WIN32
	char buffer[MAX_PATH] = { 0 };
	GetModuleFileName(NULL, buffer, MAX_PATH);
#else
	char buffer[PATH_MAX];
	ssize_t count = readlink("/proc/self/exe", buffer, PATH_MAX);
#endif
	std::string::size_type pos = std::string(buffer).find_last_of("\\/");
	return std::string(buffer).substr(0, pos);
}