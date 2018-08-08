#include <exception>


class pError : public std::exception
{
private:
	int err_;
	const char * errStr_;
public:
	pError(int err, const char * errStr) : err_(err), errStr_(errStr){}

	~pError() throw() {}

	virtual const char * what() const throw (){
		return errStr_;
	}

	int err(void) const { return err_; }
};
